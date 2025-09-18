import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full text-center space-y-6">
        {/* Illustration/Icon */}
        <div className="text-6xl mb-4">🤔</div>

        <h1 className="text-2xl font-bold">
          Hmm, we can't seem to find that page.
        </h1>

        <p className="text-muted-foreground text-lg">
          Maybe it moved, or perhaps the link was a bit off?
        </p>

        <p className="text-muted-foreground">
          Don't worry, let's get you back to familiar territory.
        </p>

        <Link href="/" className="btn btn-primary w-full max-w-xs mx-auto mt-8">
          Go Back Home
        </Link>
      </div>
    </div>
  );
}
