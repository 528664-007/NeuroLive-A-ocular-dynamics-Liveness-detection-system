"""Print-attack sample generation — NOT YET IMPLEMENTED.

A real print attack needs a physical print (or high-res photo displayed on
a static, non-refreshing surface) recorded by the actual event camera,
because the whole point of testing against it is capturing what real paper
texture / static-photo motion looks like to the sensor — that's not
something you can convincingly synthesize from RGB frames.

Once you have a handful of these recordings, wire them into
data_root/index.jsonl with attack_type="print" and they'll flow through the
existing evaluation code (see eval/metrics.py per_attack_type_breakdown)
with no other changes needed.
"""
def generate_print_attack_sample(*args, **kwargs):
    raise NotImplementedError(
        "Print-attack recordings must come from real event-camera captures of "
        "printed/displayed photos — see this file's module docstring. Nothing "
        "to synthesize here honestly; add real recordings to data_root/index.jsonl "
        "with attack_type='print' instead of calling this function."
    )
