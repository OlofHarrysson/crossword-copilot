export type NavigationLink = {
  href: string;
  label: string;
  showInNavbar: boolean; // true = navbar, false = avatar dropdown
};

const marketingNavigationLinks: NavigationLink[] = [
  { href: "/#features", label: "Features", showInNavbar: true },
  { href: "/#pricing", label: "Pricing", showInNavbar: true },
  { href: "/#faq", label: "FAQ", showInNavbar: true },
] as const;

const appNavigationLinks: NavigationLink[] = [
  { href: "/p/dashboard", label: "Dashboard", showInNavbar: true },
  // These would show in avatar dropdown
  { href: "/p/settings", label: "Settings", showInNavbar: false },
  { href: "/", label: "Homepage", showInNavbar: false },
  { href: "/logout", label: "Logout", showInNavbar: false },
] as const;

export function getNavigationLinks(
  variant: "marketing" | "app",
  section: "navbar" | "avatar" | "mobile"
): NavigationLink[] {
  const allLinks =
    variant === "app" ? appNavigationLinks : marketingNavigationLinks;

  if (section === "mobile") {
    return [...allLinks]; // All links for mobile (no distinction)
  }

  return allLinks.filter((link) =>
    section === "navbar" ? link.showInNavbar : !link.showInNavbar
  );
}
