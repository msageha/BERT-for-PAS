FROM gcr.io/jarvisml-experimental/experiment/mzk/base:latest

WORKDIR /app
ADD . ./

RUN mkdir -p /app/Downstream
RUN mkdir -p /app/Upstream

ENV DOWNSTREAM_DIR /app/Downstream
ENV UPSTREAM_DIR /app/Upstream
