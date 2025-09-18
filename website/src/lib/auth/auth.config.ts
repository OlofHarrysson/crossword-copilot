import type { NextAuthConfig } from "next-auth";
import Google from "next-auth/providers/google";
import Resend from "next-auth/providers/resend";
import NeonAdapter from "@auth/neon-adapter";
import { Pool } from "@neondatabase/serverless";

export const authConfig = (): NextAuthConfig => {
  // Create a Pool inside the request handler as recommended by Neon
  const pool = new Pool({ connectionString: process.env.DATABASE_URL });

  return {
    debug: process.env.NODE_ENV === "development",
    adapter: NeonAdapter(pool as any),
    session: {
      strategy: "database",
    },
    providers: [
      Google({
        clientId: process.env.GOOGLE_CLIENT_ID!,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      }),
      Resend({
        from: process.env.AUTH_RESEND_FROM!,
      }),
    ],
    pages: {
      signIn: "/login",
      error: "/login",
    },
    callbacks: {
      async session({ session, user }) {
        // When using database sessions, user object is available instead of token
        if (user?.id) {
          session.user.id = user.id;
        }
        return session;
      },
    },
  };
};
