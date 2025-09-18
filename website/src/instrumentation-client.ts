import posthog from "posthog-js";

posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY!, {
  // api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  api_host: "/improve-now", // Need to match your next.config.ts rewrites
  ui_host: "https://eu.posthog.com",
  defaults: "2025-05-24",
});
