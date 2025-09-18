import Link from "next/link";
import { siteConfig } from "../../site-config";
import { getSEOTags } from "@/lib/seo";

// You are an excellent lawyer.

// I need your help to write a simple Privacy Policy for my website. Here is some context:
// - Contact information: olof@tandemfuture.com
// - User data collected: name, email and payment information
// - Non-personal data collection: web cookies
// - Data storage location: EU servers
// - Data retention: As long as necessary for service provision
// - User rights: Access, correct, delete their data
// - Governing Law: Sweden

// Please write a simple Privacy Policy for my site.

const lastUpdated = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

const PrivacyText = `Privacy Policy for ${siteConfig.name}

Last updated: ${lastUpdated}

At ${
  siteConfig.name
}, we value your privacy and aim to be transparent about how we handle your information.

1. Information We Collect

- Account information: When you register, we collect your name and email
- Payment information: When you make purchases (processed securely through our payment provider)
- Usage data: Basic analytics and cookies to improve our service

2. How We Use Your Information

- To provide and maintain our services
- To process your transactions
- To send important service updates
- To improve your experience

3. Data Protection

- We store data securely on EU servers
- We only keep your data as long as necessary
- We never sell your personal information
- We use industry-standard security measures

4. Your Rights

You can:
- Access your personal data
- Update your information
- Delete your account
- Opt out of communications

5. Cookies

We use essential cookies to keep you logged in and analytics cookies to improve our service. You can control cookie settings in your browser.

6. Updates

We may update this policy occasionally. We'll notify you of significant changes.

7. Contact Us

Questions about your privacy? Contact us at ${
  siteConfig.resend.fromAdmin.match(/<(.+?)>/)?.[1] || ""
}.

This policy is governed by Swedish law.`;

export const metadata = getSEOTags({
  title: `Privacy Policy | ${siteConfig.name}`,
  keywords: ["privacy policy", "legal"],
  canonicalUrlRelative: "/privacy",
});

const Privacy = () => {
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
          Privacy Policy for {siteConfig.name}
        </h1>

        <div className="prose prose-sm sm:prose-base max-w-none">
          <p className="text-sm text-muted-foreground mb-4">
            Last Updated: {lastUpdated}
          </p>
          <div className="leading-relaxed whitespace-pre-line">
            {PrivacyText}
          </div>
        </div>
      </div>
    </main>
  );
};

export default Privacy;
