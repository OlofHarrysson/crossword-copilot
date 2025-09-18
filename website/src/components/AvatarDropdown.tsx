"use client";

import Link from "next/link";
import { signOut } from "next-auth/react";
import { NavigationLink } from "@/lib/navigation";
import { Menu } from "lucide-react";
import { siteConfig } from "@/app/site-config";

interface AvatarDropdownProps {
  avatarLinks: NavigationLink[];
}

// Used in the logged in state for desktop
export const AvatarDropdown = ({ avatarLinks }: AvatarDropdownProps) => {
  return (
    <div className="dropdown dropdown-end">
      <div
        tabIndex={0}
        role="button"
        className="btn btn-ghost"
        aria-label="Open menu"
      >
        <Menu className="h-5 w-5" />
      </div>
      <ul
        tabIndex={0}
        className="menu menu-sm dropdown-content mt-3 z-[1] p-2 shadow-lg bg-base-100 rounded-box w-52 border border-accent"
      >
        {/* Navigation Links */}
        {avatarLinks.map((link) => (
          <li key={link.href}>
            {link.href === "/logout" ? (
              <div
                onClick={() =>
                  signOut({ redirectTo: siteConfig.auth.loginUrl })
                }
                className="text-error hover:bg-error/10 cursor-pointer"
              >
                {link.label}
              </div>
            ) : (
              <Link href={link.href} className="hover:bg-accent">
                {link.label}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};
