"use client";

import { useEffect } from "react";
import Link from "next/link";

export default function Error({
  error,
}: {
  error: Error & { digest?: string };
}) {
  useEffect(() => {
    // Log the error to your error reporting service
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full text-center space-y-6">
        {/* Error Illustration */}
        <div className="text-6xl mb-4">😅</div>

        <h1 className="text-2xl font-bold">Whoops! Something went wrong.</h1>

        <p className="text-muted-foreground text-lg">
          Looks like there's a glitch on this page. Our mistake!
        </p>

        <p className="text-muted-foreground">
          Our team has been notified and will get this fixed up. Sorry about
          that!
        </p>

        <Link href="/" className="btn btn-primary w-full max-w-xs mx-auto mt-8">
          Back to Home
        </Link>
      </div>
    </div>
  );
}
