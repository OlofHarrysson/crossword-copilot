import "server-only";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth/auth";

import { siteConfig } from "@/app/site-config";

export async function getSession() {
  return await auth();
}

export async function getUser() {
  const session = await getSession();
  return session?.user ?? null;
}

export async function requireAuth() {
  const user = await getUser();
  if (!user) {
    redirect(siteConfig.auth.loginUrl);
  }
  return user;
}
