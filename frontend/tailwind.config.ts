import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        neuron: {
          bg: "#05070d",
          panel: "#0a0f1c",
          edge: "#101a2e",
          line: "#1c2a44",
        },
        signal: {
          excite: "#22d3ee", // cyan — excitatory impulse
          inhibit: "#fb7185", // rose — inhibitory gate
          info: "#a78bfa", // violet — propagation signal
          fire: "#34d399", // mint — gate pass
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      animation: {
        "pulse-fast": "pulse 0.9s ease-in-out infinite",
        "glow": "glow 1.6s ease-in-out infinite",
      },
      keyframes: {
        glow: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
