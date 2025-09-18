import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { siteConfig } from "@/app/site-config";
import { getSEOTags } from "@/lib/seo";
import SessionProvider from "@/components/providers/SessionProvider";
import "./globals.css";

const font = Inter({ subsets: ["latin"] });

// Uses default values from siteConfig
export const metadata: Metadata = getSEOTags();

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head></head>
      <body className={`${font.className} antialiased`}>
        <SessionProvider>{children}</SessionProvider>
      </body>
    </html>
  );
}
