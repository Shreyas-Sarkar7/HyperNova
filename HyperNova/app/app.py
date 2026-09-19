import json
import os
import csv
from functools import wraps
import uuid
from ps3_door.door_predict import predict_from_csv
from ps3_acv.acv_web import predict_acv
from ps3_rail.rail_web import predict_rail
from ps3_shm.shm_web import predict_shm

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(DATA_DIR, "users.json")) as f:
    USERS = {u["username"]: u["password_hash"] for u in json.load(f)["users"]}

def save_users():
    with open(os.path.join(DATA_DIR, "users.json"), "w") as f:
        json.dump(
            {"users": [{"username": u, "password_hash": h} for u, h in USERS.items()]},
            f, indent=2,
        )


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


@app.route("/")
def landing():
    if "username" in session:
        return redirect(url_for("home"))
    return render_template("landing.html")

@app.route("/home")
@login_required
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        stored_hash = USERS.get(username)
        if stored_hash and check_password_hash(stored_hash, password):
            session["username"] = username
            return redirect(request.args.get("next") or url_for("home"))
        error = "Incorrect username or password."
    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            error = "Please fill in both fields."
        elif username in USERS:
            error = "That username is already taken."
        else:
            USERS[username] = generate_password_hash(password)
            save_users()
            session["username"] = username
            return redirect(url_for("home"))
    return render_template("signup.html", error=error)


PS3_RESULTS_DIR = os.path.join(app.root_path, "static", "ps3_results")
os.makedirs(PS3_RESULTS_DIR, exist_ok=True)


@app.route("/ps3/door", methods=["GET", "POST"])
@login_required
def ps3_door():
    error = None
    results = None
    download_url = None
    n_normal = n_abnormal = 0

    if request.method == "POST":
        f = request.files.get("data_file")
        if not f or f.filename == "":
            error = "Please choose a CSV file to upload."
        elif not f.filename.lower().endswith(".csv"):
            error = "Please upload a .csv file."
        else:
            try:
                out_df = predict_from_csv(f)
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                error = f"Couldn't process that file — check it matches the Door dataset's column format. (debug: {exc})"
            else:
                filename = f"door_predictions_{uuid.uuid4().hex[:8]}.csv"
                out_df.to_csv(os.path.join(PS3_RESULTS_DIR, filename), index=False)
                download_url = url_for("static", filename=f"ps3_results/{filename}")
                results = out_df.to_dict(orient="records")
                n_normal = sum(1 for r in results if r["prediction"] == "Normal")
                n_abnormal = sum(1 for r in results if r["prediction"] == "Abnormal resistance")

    return render_template(
        "ps3_door.html", error=error, results=results, download_url=download_url,
        n_normal=n_normal, n_abnormal=n_abnormal,
    )


@app.route("/ps3/acv", methods=["GET", "POST"])
@login_required
def ps3_acv():
    error = None
    file_id = None
    ranked = None

    if request.method == "POST":
        f = request.files.get("data_file")
        if not f or f.filename == "":
            error = "Please choose a file to upload."
        elif not f.filename.lower().endswith((".xlsx", ".xls", ".csv")):
            error = "Please upload an .xlsx, .xls, or .csv file."
        else:
            try:
                from werkzeug.utils import secure_filename
                file_id, ranked = predict_acv(f, secure_filename(f.filename))
            except Exception as exc:
                error = f"Couldn't process that file — check it matches the ACV dataset's column format. (debug: {exc})"

    return render_template("ps3_acv.html", error=error, file_id=file_id, ranked=ranked)


@app.route("/ps3/rail", methods=["GET", "POST"])
@login_required
def ps3_rail():
    error = None
    results = None
    download_url = None
    counts = {}

    if request.method == "POST":
        files = [f for f in request.files.getlist("data_files") if f.filename]
        if not files:
            error = "Please choose one or more CSV files to upload."
        elif not all(f.filename.lower().endswith(".csv") for f in files):
            error = "Please upload .csv files only."
        else:
            try:
                results = predict_rail(files)
            except Exception as exc:
                error = f"Couldn't process those files — check they match the Rail Corrugation dataset's column format. (debug: {exc})"
            else:
                filename = f"rail_predictions_{uuid.uuid4().hex[:8]}.csv"
                with open(os.path.join(PS3_RESULTS_DIR, filename), "w", newline="") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=["file_id", "prediction"])
                    writer.writeheader()
                    writer.writerows(results)
                download_url = url_for("static", filename=f"ps3_results/{filename}")
                for r in results:
                    counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1

    return render_template("ps3_rail.html", error=error, results=results, download_url=download_url, counts=counts)


@app.route("/ps3/shm", methods=["GET", "POST"])
@login_required
def ps3_shm():
    error = None
    results = None
    download_url = None

    if request.method == "POST":
        files = [f for f in request.files.getlist("data_files") if f.filename]
        if not files:
            error = "Please choose one or more CSV files to upload."
        elif not all(f.filename.lower().endswith(".csv") for f in files):
            error = "Please upload .csv files only."
        else:
            try:
                results = predict_shm(files)
                for r in results:
                    r["prediction"] = round(float(r["prediction"]), 4)
            except Exception as exc:
                error = f"Couldn't process those files — check they match the SHM dataset's column format. (debug: {exc})"
            else:
                filename = f"shm_predictions_{uuid.uuid4().hex[:8]}.csv"
                with open(os.path.join(PS3_RESULTS_DIR, filename), "w", newline="") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=["file_id", "prediction"])
                    writer.writeheader()
                    writer.writerows(results)
                download_url = url_for("static", filename=f"ps3_results/{filename}")

    return render_template("ps3_shm.html", error=error, results=results, download_url=download_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
