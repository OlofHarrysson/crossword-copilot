"use client";

import { Menu, X } from "lucide-react";
import Link from "next/link";
import { AvatarDropdown } from "@/components/AvatarDropdown";
import { useAuth } from "@/lib/auth/hooks";
import { siteConfig } from "@/app/site-config";
import { getNavigationLinks, NavigationLink } from "@/lib/navigation";
import { signOut } from "next-auth/react";
import { useState } from "react";

interface NavbarProps {
  variant?: "marketing" | "app";
}

const MobileDrawer = ({ variant }: { variant: "marketing" | "app" }) => {
  const [isOpen, setIsOpen] = useState(false);
  const { isAuthenticated } = useAuth();

  const openDrawer = () => setIsOpen(true);
  const closeDrawer = () => setIsOpen(false);

  // Get all navigation links for mobile (no distinction needed)
  const allLinks = getNavigationLinks(variant, "mobile");

  const handleLinkClick = async (href: string) => {
    if (href === "/logout") {
      await signOut({ redirectTo: siteConfig.auth.loginUrl });
    }
    closeDrawer();
  };

  return (
    <>
      {/* Hamburger Menu Button */}
      <button
        onClick={openDrawer}
        className="btn btn-ghost lg:hidden"
        aria-label="Open navigation menu"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Backdrop Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={closeDrawer}
        />
      )}

      {/* Side Drawer */}
      <div
        className={`fixed top-0 right-0 h-full w-80 bg-base-100 shadow-xl z-50 transform transition-transform duration-300 ease-in-out lg:hidden ${
          isOpen ? "translate-x-0" : "translate-x-full"
        }`}
      >
        {/* Drawer Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <h2 className="text-lg font-semibold">Menu</h2>
          <button
            onClick={closeDrawer}
            className="btn btn-ghost btn-sm"
            aria-label="Close navigation menu"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="p-6">
          <ul className="space-y-2">
            {allLinks.map((link) => (
              <li key={link.href}>
                {link.href === "/logout" ? (
                  <button
                    onClick={() => handleLinkClick(link.href)}
                    className="block w-full text-left py-3 px-4 text-base hover:bg-error/10 text-error rounded-md transition-colors"
                  >
                    {link.label}
                  </button>
                ) : (
                  <Link
                    href={link.href}
                    className="block py-3 px-4 text-base hover:bg-accent hover:text-accent-foreground rounded-md transition-colors"
                    onClick={() => handleLinkClick(link.href)}
                  >
                    {link.label}
                  </Link>
                )}
              </li>
            ))}
          </ul>

          {/* Login Button for non-authenticated users */}
          {!isAuthenticated && (
            <div className="pt-6 mt-6 border-t">
              <Link
                href={siteConfig.auth.loginUrl}
                className="btn btn-primary w-full"
                onClick={closeDrawer}
              >
                Login
              </Link>
            </div>
          )}
        </nav>
      </div>
    </>
  );
};

const DesktopNav = ({ navbarLinks }: { navbarLinks: NavigationLink[] }) => {
  return (
    <ul className="menu menu-horizontal px-1">
      {navbarLinks.map((link) => (
        <li key={link.href}>
          <Link
            href={link.href}
            className="hover:bg-accent hover:text-accent-foreground"
          >
            {link.label}
          </Link>
        </li>
      ))}
    </ul>
  );
};

const Navbar = ({ variant = "marketing" }: NavbarProps) => {
  const inApp = variant === "app";
  const logoHref = inApp ? siteConfig.auth.callbackUrl : "/";

  // Get navbar and avatar links separately
  const navbarLinks = getNavigationLinks(variant, "navbar");
  const avatarLinks = getNavigationLinks(variant, "avatar");

  return (
    <div className="navbar bg-background border-b border-accent">
      <div className="navbar-start">
        <Link
          href={logoHref}
          className="btn btn-ghost text-xl hover:bg-transparent"
        >
          Your Logo
        </Link>
      </div>
      <div className="navbar-center hidden lg:flex">
        <DesktopNav navbarLinks={navbarLinks} />
      </div>
      <div className="navbar-end">
        <div className="hidden lg:flex">
          {inApp ? (
            <AvatarDropdown avatarLinks={avatarLinks} />
          ) : (
            <Link
              href={siteConfig.auth.loginUrl}
              className="btn btn-primary text-primary-foreground"
            >
              Login
            </Link>
          )}
        </div>
        <MobileDrawer variant={variant} />
      </div>
    </div>
  );
};

export default Navbar;
