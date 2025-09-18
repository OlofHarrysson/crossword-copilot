import Link from "next/link";
import Image from "next/image";
import { ExternalLink } from "lucide-react";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About",
  description:
    "Meet Olof Harrysson, an AI developer turned indie maker building digital products that solve real problems.",
};

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 sm:px-6 py-8 sm:py-12 max-w-2xl">
        <div className="flex flex-col items-center text-center">
          <Image
            src="/assets/profile.jpg"
            alt="Olof Harrysson"
            width={200}
            height={200}
            className="rounded-full mb-4 sm:mb-6 w-40 h-40 sm:w-50 sm:h-50"
            priority
          />
          <p className="text-sm sm:text-base text-muted-foreground mb-2 sm:mb-4">
            This project is created by
          </p>
          <h1 className="text-2xl sm:text-3xl font-bold text-foreground mb-6 sm:mb-8">
            Olof Harrysson
          </h1>
        </div>

        <div className="text-center mb-6 sm:mb-8">
          <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
            After 6 years as an AI developer, I'm now building my own digital
            products. I'm working solo, building micro-projects that solve
            specific problems for specific people.
          </p>
        </div>

        <div className="text-center">
          <Link
            href="https://tandemfuture.com"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 btn btn-primary text-sm sm:text-base px-4 sm:px-6 py-2 sm:py-3"
          >
            Learn more about me
            <ExternalLink className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </div>
  );
}
