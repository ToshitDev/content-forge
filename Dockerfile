# python:3.12-slim: a real Debian userland (so pip can build anything
# that needs it) without the extra compilers/docs/tooling the full
# python:3.12 image ships with — smaller image, same Python.
FROM python:3.12-slim

# Don't buffer stdout/stderr — without this, print()/logging output can
# sit in a buffer instead of showing up in `docker logs` right away.
ENV PYTHONUNBUFFERED=1

# Everything the app needs lives under /app inside the container.
WORKDIR /app

# Copy only requirements.txt first, then install. Docker caches each
# layer by its inputs — as long as requirements.txt doesn't change,
# rebuilding after an app.py edit reuses this layer instead of
# reinstalling every dependency from scratch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project. This layer invalidates on nearly
# every change, but it's cheap — no dependency resolution here, just a
# file copy.
COPY . .

# Streamlit's default port.
EXPOSE 8501

# --server.address=0.0.0.0 is required, not optional: Streamlit binds
# to localhost by default, and "localhost" inside the container is the
# container itself, not the host machine — without this flag, the port
# mapping in docker-compose.yml would have nothing reachable behind it.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
