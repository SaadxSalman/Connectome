import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "SynapseCraft · Connectome Cockpit",
  description:
    "A biologically-inspired agentic RAG engine — watch queries propagate as electrical impulses across a simulated neural wiring diagram.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neuron-bg font-mono text-sm antialiased">
        {children}
      </body>
    </html>
  );
}
