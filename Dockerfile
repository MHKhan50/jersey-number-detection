FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

ENTRYPOINT ["jersey-detect"]
CMD ["--help"]
