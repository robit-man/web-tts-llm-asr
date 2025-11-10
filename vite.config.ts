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
      },
    },
  ],
  assetsInclude: ["**/*.wasm"],
});
