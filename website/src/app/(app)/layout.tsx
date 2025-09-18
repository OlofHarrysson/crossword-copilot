import { BaseLayout } from "@/components/layouts/BaseLayout";
import Navbar from "@/components/Navbar";

export default async function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <BaseLayout navbar={<Navbar variant="app" />}>{children}</BaseLayout>;
}
