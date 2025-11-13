import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

// https://vite.dev/config/
export default defineConfig({
  worker: {
    format: "es",
  },
  optimizeDeps: {
    exclude: ["@mlc-ai/web-llm"],
  },
  ssr: {
    noExternal: ["@mlc-ai/web-llm"],
  },
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      devOptions: {
        enabled: true,
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,ico,png,svg,wasm,json}"],
        maximumFileSizeToCacheInBytes: 35 * 1024 * 1024,
      },
      includeAssets: ["vite.svg"],
      manifest: {
        name: "Trifecta Voice Lab",
        short_name: "Trifecta",
        description:
          "Unified Whisper ASR, WebLLM reasoning, and Piper TTS as a single-page PWA demo.",
        theme_color: "#0f172a",
        background_color: "#020617",
        start_url: "/",
        display: "standalone",
        icons: [
          {
            src: "icons/icon-192.png",
            sizes: "192x192",
            type: "image/png",
            purpose: "any maskable",
          },
          {
            src: "icons/icon-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "any maskable",
          },
        ],
      },
    }),
    {
      name: "onnx-runtime-dev-server",
      configureServer(server) {
        server.middlewares.use("/onnx-runtime", (req, res, next) => {
          if (req.url && req.url.includes("?import")) {
            req.url = req.url.replace("?import", "");
          }
          if (req.url && req.url.endsWith(".mjs")) {
            res.setHeader("Content-Type", "application/javascript");
            res.setHeader("Access-Control-Allow-Origin", "*");
          }
          next();
        });

        server.middlewares.use("/tts-model", (_req, res, next) => {
          res.setHeader("Cache-Control", "public, max-age=604800, immutable");
          res.setHeader("ETag", '"model-v1"');
          next();
        });

        // CORS proxy for fetching external models
        server.middlewares.use("/cors-proxy", async (req, res) => {
          const url = new URL(req.url || "", `http://${req.headers.host}`);
          const targetUrl = url.searchParams.get("url");

          if (!targetUrl) {
            res.statusCode = 400;
            res.end("Missing url parameter");
            return;
          }

          try {
            const response = await fetch(targetUrl);

            // Set CORS headers
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
            res.setHeader("Access-Control-Allow-Headers", "*");

            // Copy content type from original response
            const contentType = response.headers.get("content-type");
            if (contentType) {
              res.setHeader("Content-Type", contentType);
            }

            // Stream the response
            const buffer = await response.arrayBuffer();
            res.end(Buffer.from(buffer));
          } catch (error) {
            console.error("CORS proxy error:", error);
            res.statusCode = 500;
            res.end(`Error fetching resource: ${(error as Error).message}`);
          }
        });
      },
    },
  ],
  assetsInclude: ["**/*.wasm"],
});
