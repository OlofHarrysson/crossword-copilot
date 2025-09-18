import { Toaster } from "react-hot-toast";
import Footer from "@/components/Footer";

interface BaseLayoutProps {
  children: React.ReactNode;
  navbar: React.ReactNode;
}

export function BaseLayout({ children, navbar }: BaseLayoutProps) {
  return (
    <>
      {navbar}
      <main>{children}</main>
      <Footer />
      <Toaster
        position="bottom-right"
        toastOptions={{
          className: "!rounded-md !p-4 !shadow-md",
          success: {
            className: "toast-success",
            duration: 4000,
          },
          error: {
            className: "toast-error",
            duration: 4000,
          },
        }}
      />
    </>
  );
}
