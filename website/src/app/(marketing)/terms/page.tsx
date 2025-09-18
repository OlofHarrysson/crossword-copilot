import Link from "next/link";
import { siteConfig } from "../../site-config";
import { getSEOTags } from "@/lib/seo";

// You are an excellent lawyer.

// I need your help to write a simple Terms & Services for my website. Here is some context:
// - Contact information: olof@tandemfuture.com
// - User data collected: name, email and payment information
// - Non-personal data collection: web cookies
// - Governing Law: Sweden
// - Updates to the Terms: users will be updated by email

// Please write a simple Terms & Services for my site.

const lastUpdated = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

const TermsText = `Terms of Service for ${siteConfig.name}

These Terms of Service ("Terms") govern your use of ${
  siteConfig.name
} ("we", "us", or "our") website at ${siteConfig.domain} ("Website").

1. Acceptance of Terms

By accessing or using our Website, you agree to be bound by these Terms. If you do not agree to these Terms, please do not use our Website.

2. User Accounts

- You must provide accurate and complete information when creating an account
- You are responsible for maintaining the security of your account
- You must notify us immediately of any unauthorized access to your account
- We reserve the right to suspend or terminate accounts that violate these Terms

3. User Conduct

You agree not to:
- Violate any applicable laws or regulations
- Interfere with the proper functioning of the Website
- Attempt to gain unauthorized access to our systems
- Use the Website for any illegal or unauthorized purpose

4. Intellectual Property

All content on the Website, including but not limited to text, graphics, logos, and software, is our property and is protected by intellectual property laws.

5. Payment Terms

- Payment information is processed securely through our payment providers
- All fees are non-refundable unless otherwise stated
- We reserve the right to modify our pricing with reasonable notice

6. Limitation of Liability

To the maximum extent permitted by law, we shall not be liable for any indirect, incidental, special, consequential, or punitive damages.

7. Termination

We reserve the right to terminate or suspend access to our Website without prior notice for violations of these Terms.

8. Changes to Terms

We may modify these Terms at any time. Continued use of the Website after changes constitutes acceptance of the modified Terms.

9. Governing Law

These Terms are governed by the laws of Sweden. Any disputes shall be resolved in Swedish courts.

10. Contact Information

For questions about these Terms, please contact us at ${
  siteConfig.resend.fromAdmin.match(/<(.+?)>/)?.[1] || ""
}.`;

export const metadata = getSEOTags({
  title: `Terms of Service | ${siteConfig.name}`,
  keywords: ["terms of service", "legal"],
  canonicalUrlRelative: "/terms",
});

const Terms = () => {
  return (
    <main className="max-w-xl mx-auto">
      <div className="p-5">
        <Link href="/" className="btn btn-ghost">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="w-5 h-5"
          >
            <path
              fillRule="evenodd"
              d="M15 10a.75.75 0 01-.75.75H7.612l2.158 1.96a.75.75 0 11-1.04 1.08l-3.5-3.25a.75.75 0 010-1.08l3.5-3.25a.75.75 0 111.04 1.08L7.612 9.25h6.638A.75.75 0 0115 10z"
              clipRule="evenodd"
            />
          </svg>
          Back
        </Link>
        <h1 className="text-3xl font-extrabold pb-6">
          Terms of Service for {siteConfig.name}
        </h1>

        <div className="prose prose-sm sm:prose-base max-w-none">
          <p className="text-sm text-muted-foreground mb-4">
            Last Updated: {lastUpdated}
          </p>
          <div className="leading-relaxed whitespace-pre-line">{TermsText}</div>
        </div>
      </div>
    </main>
  );
};

export default Terms;
