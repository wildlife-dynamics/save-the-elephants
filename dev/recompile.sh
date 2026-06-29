#!/bin/bash

workflow=$1
shift
flags=$*

pixi update --manifest-path pixi.toml -e compile

# wt-compiler 0.8.0 uses PEP 695 `type` alias syntax which pydantic <2.9 can't handle.
# ecoscope-platform pins pydantic <2.9, so we patch the installed files after every update.
WT_SPEC=".pixi/envs/compile/lib/python3.12/site-packages/wt_compiler/spec.py"
sed -i '' 's/^type PartialKwargs = /PartialKwargs = /' "$WT_SPEC"

# wt-compiler 0.8.0 uses ChannelPriority.Strict in its rattler solve() call, which excludes
# pydeck 0.9.2 from conda-forge when the prefix.dev channels have a different pydeck version.
# Patch to ChannelPriority.Disabled so the solver can pick the best match from any channel.
WT_DISCOVERY=".pixi/envs/compile/lib/python3.12/site-packages/wt_compiler/discovery.py"
pixi run --manifest-path pixi.toml -e compile python3 - <<'PYEOF'
import pathlib

p = pathlib.Path(".pixi/envs/compile/lib/python3.12/site-packages/wt_compiler/discovery.py")
content = p.read_text()

content = content.replace(
    "from rattler import Channel, MatchSpec, Platform, VirtualPackage, install, solve",
    "from rattler import Channel, ChannelPriority, MatchSpec, Platform, VirtualPackage, install, solve",
)
content = content.replace(
    "            virtual_packages=virtual_packages,\n        )",
    "            virtual_packages=virtual_packages,\n            channel_priority=ChannelPriority.Disabled,\n        )",
)
p.write_text(content)
PYEOF

# (re)initialize dot executable to ensure graphviz is available
pixi run --manifest-path pixi.toml -e compile dot -c

echo "recompiling workflows/${workflow}/spec.yaml with flags '--clobber ${flags}'"

command="pixi run --manifest-path pixi.toml -e compile \
wt-compiler compile --spec workflows/${workflow}/spec.yaml --clobber ${flags}"

exec $command
