**Trackside HQ — Predictive Monitoring for SMRT Rail Operations**

Trackside HQ is an end-to-end monitoring system for SMRT rail operations, built to make maintenance and continued operation simpler, faster, and more proactive. It ingests raw sensor data from four subsystems — structural health, rail corrugation, train doors, and ventilation (ACV) — compresses each stream into a compact feature set, and reduces the result to a simple, colour-coded **safe / concern / danger** signal that a duty officer can read at a glance.

---

**Part 1 — Structural Health Monitoring (SHM)**

*Input:* Continuous dynamic-stress time-series recorded from load-bearing structures (carbodies, bogie frames) during normal service. Each file is a one-second CSV segment from a measurement point, sampled from the vehicle's onboard stress sensors.

*Output:* A single numeric **cumulative fatigue damage** value per file, computed via rainflow cycle counting and Miner's linear damage rule, refined by a RandomForest regressor and a fitted power-law model that compete under cross-validation. The winning model is saved and reused, mapping each new file to a safe / concern / danger band.

---

**Part 2 — Rail Corrugation**

*Input:* One-second axle-box vibration and shock recordings at 10 kHz from 64 accelerometers across 8 cars, plus a rotational-speed channel. Each file is judged as a whole, but Side I and Side II rails must be assessed independently.

*Output:* A three-class label per file — **Normal**, **Side I**, or **Side II** corrugation — produced by a classifier trained on frequency-domain and statistical features extracted from the vibration channels. The dashboard highlights the affected side and segment on the network map so maintenance crews know exactly where to grind.

---

**Part 3 — Train Door Health**

*Input:* A single continuous stream of door-controller telemetry — motor current, voltage, back-EMF, door position, and open/close flags — running through many door cycles back to back with irregular gaps, the way a live onboard system actually produces it.

*Output:* One row per detected door cycle, giving its **start time**, **end time**, and a **Normal** or **Abnormal resistance** classification. The pipeline segments the stream itself (no pre-cut examples), then classifies each cycle, outputting a downloadable per-cycle CSV for engineering follow-up.

---

**Part 4 — Ventilation / ACV System**

*Input:* Multivariate 30-second telemetry from all 8 cars of a train — pressures, temperatures, running modes, control modes, and self-check statuses. Column sets vary between files, so the loader reads each file's own headers.

*Output:* A **ranked list of cars**, most-to-least likely to have a refrigerant leak, using the car identifier exactly as it appears in the file's own headers. A linear rank-decay scoring rewards correctly narrowing down to a short list even when the top pick isn't exactly right — matching real diagnostic practice.

---

Underneath all four sits a shared fatigue-analysis pipeline: rainflow counting, Miner's-rule damage accumulation, and a lightweight machine-learning layer. The system installs its own dependencies and trains itself on first launch, so any operator can run it out of the box. By turning raw telemetry into a single intuitive safe / concern / danger signal, Trackside HQ gives SMRT the clarity to act before faults become failures — and the confidence to keep trains running safely, reliably, and on schedule.