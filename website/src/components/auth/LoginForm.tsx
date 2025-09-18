"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import GoogleSignInButton from "@/components/auth/GoogleSignInButton";
import { siteConfig } from "@/app/site-config";

interface LoginFormProps {
  error?: string;
}

export default function LoginForm({ error }: LoginFormProps) {
  const [email, setEmail] = useState("");
  const [isEmailLoading, setIsEmailLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsEmailLoading(true);

    try {
      const result = await signIn("resend", {
        email,
        redirect: false,
      });

      if (!result?.error) {
        setEmailSent(true);
      }
    } catch (error) {
      console.error("Email sign-in failed:", error);
    } finally {
      setIsEmailLoading(false);
    }
  };

  const getErrorMessage = (error: string) => {
    switch (error) {
      case "OAuthSignin":
      case "OAuthCallback":
      case "OAuthCreateAccount":
      case "EmailCreateAccount":
      case "Callback":
        return "Google sign-in failed. Please try again.";
      case "OAuthAccountNotLinked":
        return "Account already exists with different provider.";
      case "EmailSignin":
        return "Unable to send magic link. Please check your email address and try again.";
      case "Verification":
        return "Invalid or expired magic link. Please request a new one.";
      case "SessionRequired":
        return "Please sign in to access this page.";
      default:
        return "Something went wrong. Please try again.";
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-base-100 px-4">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-base-content">
            Sign in to your account
          </h2>
          <p className="mt-2 text-center text-sm text-base-content/60">
            Choose your preferred sign-in method
          </p>
        </div>

        {/* URL Error Display */}
        {error && (
          <div className="alert alert-error">
            <span>{getErrorMessage(error)}</span>
          </div>
        )}

        {/* Google OAuth Button */}
        <div
          onClick={() =>
            signIn("google", { redirectTo: siteConfig.auth.callbackUrl })
          }
        >
          <GoogleSignInButton />
        </div>

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-base-300" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-base-100 text-base-content">
              Or sign in with email
            </span>
          </div>
        </div>

        {/* Email Sign-in Form */}
        <form onSubmit={handleEmailSubmit} className="mt-8 space-y-6">
          <div>
            <label htmlFor="email" className="sr-only">
              Email address
            </label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="input input-bordered w-full"
              placeholder="Enter your email address"
              disabled={isEmailLoading || emailSent}
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary w-full"
            disabled={isEmailLoading || emailSent}
          >
            {isEmailLoading ? (
              <>
                <span className="loading loading-spinner loading-sm"></span>
                Sending...
              </>
            ) : emailSent ? (
              "Email Sent"
            ) : (
              "Send Sign-in Link"
            )}
          </button>

          <p className="text-xs text-center text-base-content/60">
            {emailSent
              ? "Check your email for the sign-in link"
              : "We'll email you a secure link to sign in instantly"}
          </p>
        </form>
      </div>
    </div>
  );
}
