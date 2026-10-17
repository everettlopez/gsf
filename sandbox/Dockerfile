# The sandbox the agents' hands run commands in (src/workspace.py, DockerRunner):
# Python 3.12 for the build and its tests, Node 22 for the site's JavaScript tests
# (node --test). Nothing else: the container runs with no network, so whatever the
# site needs has to be written, not installed.
#
# Built automatically the first time it is needed, or by hand:
#     docker build -t gsf-sandbox:py3.12-node22 sandbox
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && node --version && python --version
WORKDIR /work
