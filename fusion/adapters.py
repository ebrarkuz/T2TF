"""Mevcut fuzyon cekirdeklerini ortak urun arayuzune uyarlar."""

from __future__ import annotations

from typing import Any

from .algorithms import advanced_ca, basic_cv, dual_imm


def _set_parameters(module, parameters: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(
            f"Desteklenmeyen parametre(ler): {', '.join(sorted(unknown))}. "
            f"Desteklenenler: {', '.join(sorted(allowed))}"
        )
    for name, value in parameters.items():
        if name in {"use_ci", "adaptive_q_enabled", "gate_threshold"}:
            continue
        setattr(module, name, value)


def _snapshot(track, indices, *, used=False, stage="", filter_name="", model=None):
    px, vx, py, vy, pz, vz = indices
    probabilities = {}
    if hasattr(track, "xy_mode_prob"):
        probabilities = {str(k): float(v) for k, v in track.xy_mode_prob.items()}
        probabilities["CT"] = probabilities.get("CT_LEFT", 0.0) + probabilities.get("CT_RIGHT", 0.0)
    status = getattr(track, "status", getattr(track, "state_status", "TENTATIVE"))
    return {
        "track_id": str(track.id),
        "track_status": str(status).lower(),
        "state": {
            "x_m": float(track.state[px, 0]), "vx_mps": float(track.state[vx, 0]),
            "y_m": float(track.state[py, 0]), "vy_mps": float(track.state[vy, 0]),
            "z_m": float(track.state[pz, 0]), "vz_mps": float(track.state[vz, 0]),
        },
        "position_variance": [float(track.cov[i, i]) for i in (px, py, pz)],
        "velocity_variance": [float(track.cov[i, i]) for i in (vx, vy, vz)],
        "filter_name": filter_name,
        "dominant_model": model,
        "model_probabilities": probabilities,
        "measurement_used": bool(used),
        "association_stage": stage,
    }


class BasicCVAdapter:
    allowed_parameters = {"gate_threshold", "DUPLICATE_DIST_M", "DUPLICATE_VEL_MPS"}

    def __init__(self, parameters: dict[str, Any]):
        _set_parameters(basic_cv, parameters, self.allowed_parameters)
        self.parameters = dict(parameters)
        self.gate_threshold = float(parameters.get("gate_threshold", 25.0))
        self.reset()

    def reset(self) -> None:
        basic_cv.GlobalTrack._id_counter = 0
        self.center = basic_cv.FusionCenter()
        self.processed = 0
        self.last_assignment = None

    def process_measurement(self, measurement):
        before = {t.id: t.hits_count for t in self.center.global_tracks}
        self.center.process_measurement(
            measurement["timestamp"], measurement["state"], measurement["cov"],
            measurement["tq"], gate_threshold=self.gate_threshold,
        )
        assigned = None
        for track in self.center.global_tracks:
            if track.id not in before or track.hits_count > before.get(track.id, -1):
                assigned = track.id
                break
        self.processed += 1
        self.last_assignment = assigned
        return [
            _snapshot(t, (0, 1, 2, 3, 4, 5), used=t.id == assigned,
                      stage="update_or_create" if t.id == assigned else "",
                      filter_name="Basic CV", model="CV")
            for t in self.center.global_tracks
        ]

    def get_diagnostics(self):
        return {"algorithm": "basic_cv", "processed_measurements": self.processed,
                "active_track_count": len(self.center.global_tracks),
                "last_assigned_track_id": self.last_assignment}

    @property
    def tracks(self):
        return self.center.global_tracks


class AdvancedCAAdapter:
    allowed_parameters = {
        "use_ci", "adaptive_q_enabled", "PROCESS_NOISE_INTENSITY",
        "BASE_GATE_CHI2_6DOF", "GATE_SCALE_MIN", "GATE_SCALE_MAX",
        "MAX_ASSOC_VEL_DIFF_MPS", "MAX_ASSOC_ACCEL_MPS2",
        "MAX_POSITION_INNOVATION_M", "COAST_TIME_LIMIT", "CONFIRM_HITS",
        "MIN_CONFIRM_AGE_S", "MIN_CONFIRM_SENSORS", "DUPLICATE_DIST_M",
        "DUPLICATE_VEL_MPS", "DUPLICATE_TIME_S",
    }

    def __init__(self, parameters: dict[str, Any]):
        _set_parameters(advanced_ca, parameters, self.allowed_parameters)
        self.parameters = dict(parameters)
        self.reset()

    def reset(self) -> None:
        advanced_ca.GlobalTrack._cnt = 0
        self.center = advanced_ca.FusionCenter(
            use_ci=bool(self.parameters.get("use_ci", True)),
            adaptive_q_enabled=bool(self.parameters.get("adaptive_q_enabled", True)),
        )
        self.processed = 0
        self.last_assignment = None

    def process_measurement(self, measurement):
        src = measurement["src"]
        before = set(self.center.src_map)
        self.center.process_batch(measurement["timestamp"], [measurement])
        assigned = self.center.src_map.get(src)
        stage = "source_map" if src in before else "association_or_create"
        self.processed += 1
        self.last_assignment = assigned
        return [
            _snapshot(t, (0, 1, 3, 4, 6, 7), used=t.id == assigned, stage=stage,
                      filter_name="Advanced CA", model="CA")
            for t in self.center.tracks
        ]

    def get_diagnostics(self):
        return {"algorithm": "advanced_ca", "processed_measurements": self.processed,
                "active_track_count": len(self.center.tracks),
                "last_assigned_track_id": self.last_assignment}

    @property
    def tracks(self):
        return self.center.tracks


class DualIMMAdapter:
    allowed_parameters = {
        "use_ci", "GATE_CONFIDENCE", "GATE_CHI2_4DOF", "GATE_CHI2_2DOF",
        "MANEUVER_GATE_MULTIPLIER", "CV_Z_PROCESS_NOISE_INTENSITY",
        "SINGER_TAU_Z", "SINGER_SIGMA_Z", "CV_XY_PROCESS_NOISE_INTENSITY",
        "CA_XY_PROCESS_NOISE_INTENSITY", "CT_XY_PROCESS_NOISE_INTENSITY",
        "COAST_TIME_LIMIT", "CONFIRM_HITS", "CONFIRM_EXISTENCE_THRESHOLD",
        "MIN_CONFIRM_RADARS", "SINGLE_RADAR_CONFIRM_HITS",
        "SINGLE_RADAR_CONFIRM_EXISTENCE", "SOURCE_MAP_TIMEOUT_S",
        "REVIVE_WINDOW_S", "REVIVE_DIST_XY_M", "REVIVE_DIST_Z_M",
        "REVIVE_VEL_MPS", "DUPLICATE_DIST_XY_M", "DUPLICATE_DIST_Z_M",
        "DUPLICATE_VEL_MPS",
    }

    def __init__(self, parameters: dict[str, Any]):
        _set_parameters(dual_imm, parameters, self.allowed_parameters)
        if "GATE_CONFIDENCE" in parameters:
            confidence = float(parameters["GATE_CONFIDENCE"])
            if not 0.0 < confidence < 1.0:
                raise ValueError("GATE_CONFIDENCE 0 ile 1 arasinda olmali")
            if "GATE_CHI2_4DOF" not in parameters:
                dual_imm.GATE_CHI2_4DOF = float(dual_imm.chi2.ppf(confidence, df=4))
            if "GATE_CHI2_2DOF" not in parameters:
                dual_imm.GATE_CHI2_2DOF = float(dual_imm.chi2.ppf(confidence, df=2))
        self.parameters = dict(parameters)
        self.reset()

    def reset(self) -> None:
        dual_imm.GlobalTrackIMM3._cnt = 0
        self.center = dual_imm.FusionCenterIMM3(
            use_ci=bool(self.parameters.get("use_ci", True)), verbose=False,
            debug=False, collect_diagnostics=True,
        )
        self.processed = 0
        self.last_assignment = None
        self.last_stage = ""

    def process_measurement(self, measurement):
        start = len(self.center.diagnostic_events)
        self.center.process_batch(measurement["timestamp"], [measurement])
        events = [e for e in self.center.diagnostic_events[start:]
                  if e.get("event_type") == "measurement"]
        event = events[-1] if events else {}
        self.center.diagnostic_events.clear()
        assigned = event.get("assigned_global_track_id")
        stage = event.get("accepted_stage", "")
        self.processed += 1
        self.last_assignment, self.last_stage = assigned, stage
        return [
            _snapshot(t, (0, 1, 3, 4, 6, 7), used=str(t.id) == str(assigned),
                      stage=stage if str(t.id) == str(assigned) else "",
                      filter_name="Dual-IMM", model=t.dominant_model())
            for t in self.center.tracks
        ]

    def get_diagnostics(self):
        return {"algorithm": "dual_imm", "processed_measurements": self.processed,
                "active_track_count": len(self.center.tracks),
                "last_assigned_track_id": self.last_assignment,
                "last_association_stage": self.last_stage}

    @property
    def tracks(self):
        return self.center.tracks
