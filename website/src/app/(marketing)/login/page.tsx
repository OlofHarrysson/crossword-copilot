import { redirect } from "next/navigation";
import { auth } from "@/lib/auth/auth";
import { siteConfig } from "@/app/site-config";
import LoginForm from "@/components/auth/LoginForm";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  // Check if user is already authenticated on the server
  const session = await auth();

  // If authenticated, redirect to dashboard
  if (session) {
    redirect(siteConfig.auth.callbackUrl);
  }

  // Await searchParams before accessing its properties
  const params = await searchParams;

  // Otherwise, show the login form
  return <LoginForm error={params.error} />;
}
