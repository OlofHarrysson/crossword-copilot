import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["node-appwrite"],
  // Add PostHog rewrites/proxy. Note: improve-now is a random string to avoid adblocks
  async rewrites() {
    return [
      {
        source: "/improve-now/static/:path*",
        destination: "https://eu-assets.i.posthog.com/static/:path*",
      },
      {
        source: "/improve-now/:path*",
        destination: "https://eu.i.posthog.com/:path*",
      },
    ];
  },
  // This is required to support PostHog trailing slash API requests
  skipTrailingSlashRedirect: true,
};

export default nextConfig;
