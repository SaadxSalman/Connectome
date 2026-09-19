"use client";

/** Live neural-activity WebSocket: auto-reconnects, replays missed spikes. */

import { useEffect, useRef, useState } from "react";

import { wsUrl } from "@/lib/api";
import type { NeuralEvent } from "@/lib/types";

export type LinkState = "connecting" | "live" | "down";

export function useNeuralStream(limit = 220) {
  const [events, setEvents] = useState<NeuralEvent[]>([]);
  const [link, setLink] = useState<LinkState>("connecting");
  const [pulse, setPulse] = useState(0); // increments on every spike (visual ticker)
  const wsRef = useRef<WebSocket | null>(null);
  const backoff = useRef(1000);

  useEffect(() => {
    let closed = false;

    const connect = () => {
      if (closed) return;
      setLink((s) => (s === "live" ? s : "connecting"));
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        backoff.current = 1000;
        setLink("live");
      };
      ws.onmessage = (msg) => {
        try {
          const evt = JSON.parse(msg.data) as NeuralEvent;
          if (evt.type === "heartbeat") return;
          setEvents((prev) => [...prev.slice(-(limit - 1)), evt]);
          if (evt.type !== "hello") setPulse((p) => p + 1);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        if (closed) return;
        setLink("down");
        backoff.current = Math.min(backoff.current * 2, 10000);
        setTimeout(connect, backoff.current);
      };
      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      closed = true;
      wsRef.current?.close();
    };
  }, [limit]);

  return { events, link, pulse };
}
