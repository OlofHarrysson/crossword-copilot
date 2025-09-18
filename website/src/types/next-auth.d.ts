import { type DefaultSession } from "next-auth";

// Extends NextAuth session to include user ID
declare module "next-auth" {
  interface Session {
    user: {
      id: string;
    } & DefaultSession["user"];
  }
}
