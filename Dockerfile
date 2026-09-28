FROM oven/bun:1.4.2
WORKDIR /app
COPY server.ts classify.js index.html ./
USER bun
ENV PORT=3000
EXPOSE 3000
CMD ["bun", "run", "server.ts"]
