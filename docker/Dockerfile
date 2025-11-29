FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# --- System deps ---
RUN apt-get update && apt-get install -y \
    build-essential git curl wget unzip \
    python3 python3-pip python3-venv \
    libssl-dev libz-dev libffi-dev \
    apt-transport-https ca-certificates gnupg \
    tree \
    && rm -rf /var/lib/apt/lists/*

# --- Install .NET 8 (for Dafny) ---
RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && apt-get update \
    && apt-get install -y dotnet-sdk-8.0 \
    && rm packages-microsoft-prod.deb

# --- Install Dafny 4.11.0 ---
RUN wget https://github.com/dafny-lang/dafny/releases/download/v4.11.0/dafny-4.11.0-x64-ubuntu-22.04.zip \
    && unzip dafny-4.11.0-x64-ubuntu-22.04.zip -d /opt/dafny \
    && rm dafny-4.11.0-x64-ubuntu-22.04.zip

ENV DAFNY_PATH=/opt/dafny/dafny
ENV PATH="$DAFNY_PATH:$PATH"

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx

# --- Cache dir for uv ---
VOLUME /root/.cache/uv

# --- Fix interpreter mapping & precreate clean venv ---
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create the venv explicitly with Python
RUN /usr/bin/python3 -m venv /workspace/.venv

# Tell uv to respect system python inside container
ENV UV_LINK_MODE=copy
ENV UV_NO_SYNC_VENV=1

WORKDIR /workspace
CMD ["/bin/bash"]
