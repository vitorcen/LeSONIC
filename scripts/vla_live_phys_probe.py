"""Composing eval-callback: drive G1 with LIVE MaskBeT tokens AND measure the real physical
fall in the SAME step — for the P1b closed-loop survival screen.

WHY a composer: the eval loop runs callbacks via ``all(cb.eval_step(...) for cb in cbs)``, which
SHORT-CIRCUITS on the first falsy return. VlaLiveInjector.eval_step always returns False ("never
exit"), so a second [vla_live, phys_valid] callback's eval_step is NEVER reached — phys_valid
never inits, never dumps. This wraps both: inject first (its False is swallowed), then measure;
return the phys verdict so the loop dumps + exits cleanly when the window completes.

Run with the 4 strict deviation terminations nulled (so the robot plays the window to the end and
phys_valid logs the REAL fall, not an envelope trip) + motion_time_out left on.
"""
from __future__ import annotations

from gear_sonic.data.vla_live_injector import VlaLiveInjector
from gear_sonic.data.phys_valid_screen import PhysValidScreen


class VlaLivePhysProbe:
    model = None  # eval loop injects the live WBC model here; forwarded to both children

    def __init__(self, host: str = "127.0.0.1", port: int = 5557, action_horizon: int = 40,
                 timeline_json: str = "", output_dir: str = "/tmp/sonic_p1b",
                 root_z_fall: float = 0.6, tilt_fall_deg: float = 40.0):
        self.vla = VlaLiveInjector(host=host, port=port, action_horizon=action_horizon,
                                   timeline_json=timeline_json)
        self.phys = PhysValidScreen(output_dir=output_dir, root_z_fall=root_z_fall,
                                    tilt_fall_deg=tilt_fall_deg)

    def on_step_end(self, *args, **kwargs):
        for child in (self.vla, self.phys):
            fn = getattr(child, "on_step_end", None)
            if callable(fn):
                fn(*args, **kwargs)

    def eval_step(self, env, results) -> bool:
        self.vla.eval_step(env, results)         # inject one live MaskBeT token (returns False)
        return self.phys.eval_step(env, results)  # measure fall; True -> dump JSON + exit loop
