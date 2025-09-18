import { siteConfig } from "../site-config";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center w-full">
      {/* Hero Section - Band 1 (White Background) */}
      <section className="relative flex flex-col items-center justify-center w-full p-4 sm:p-8 min-h-[85vh] bg-background overflow-hidden">
        <div className="relative max-w-4xl w-full space-y-8 text-center z-10">
          <div className="space-y-4">
            <div className="inline-block px-4 py-2 bg-primary/10 rounded-full text-primary font-medium text-sm border border-primary/20">
              ✨ Welcome to the future
            </div>
            <h1 className="text-3xl sm:text-6xl font-bold space-y-2">
              <div>Welcome to</div>
              <div>
                <span className="relative inline-block">
                  <span className="text-2xl sm:text-5xl relative z-10 text-primary-foreground">
                    {siteConfig.name}
                  </span>
                  <span className="absolute inset-0 bg-primary transform -rotate-1 rounded-sm -m-2"></span>
                </span>
              </div>
            </h1>
          </div>
          <p className="text-xl sm:text-2xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
            {siteConfig.description} - built for people who want to{" "}
            <span className="relative inline-block">
              get things done
              <span className="absolute -bottom-1 left-0 w-full h-1 bg-primary transform -rotate-2 origin-left"></span>
            </span>{" "}
            without the complexity.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8">
            <Link
              href={siteConfig.auth.loginUrl}
              className="btn btn-primary btn-lg shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Get Started
            </Link>
            <a
              href="#features"
              className="btn btn-outline btn-lg hover:shadow-lg transform hover:scale-105"
            >
              Learn More ↓
            </a>
          </div>
        </div>
      </section>

      {/* Problem Section - Band 2 (Muted Background) */}
      <section className="w-full bg-muted py-20 px-4 sm:px-8">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-4">
            <h2 className="text-4xl sm:text-5xl font-bold">
              💭 The Problem We Solve
            </h2>
            <p className="text-xl text-muted-foreground">
              Understanding the challenges you face and why it matters
            </p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-medium text-primary">
              Brief description of the problem your app solves and{" "}
              <span className="font-extrabold">why it matters</span> to you.
            </p>
          </div>
        </div>
      </section>

      {/* Features Section - Band 3 (White Background) */}
      <section
        id="features"
        className="w-full bg-background py-20 px-4 sm:px-8"
      >
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-4">
            <h2 className="text-4xl sm:text-5xl font-bold">✨ Key Features</h2>
            <p className="text-xl text-muted-foreground">
              Everything you need to succeed
            </p>
          </div>

          <div className="grid gap-8 sm:grid-cols-1 max-w-3xl mx-auto">
            {["Feature 1", "Feature 2", "Feature 3"].map((feature, i) => (
              <div
                key={i}
                className="bg-muted/30 p-4 sm:p-8 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-muted"
              >
                <div className="flex items-center gap-4 mb-4">
                  <span className="text-3xl">🚀</span>
                  <h3 className="text-xl font-semibold">{feature}</h3>
                </div>
                <p className="text-lg text-muted-foreground">
                  Feature description goes here. Explain how this feature helps
                  your users achieve their goals.
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section - Band 4 (Muted Background) */}
      <section id="pricing" className="w-full bg-muted py-20 px-4 sm:px-8">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-4">
            <h2 className="text-4xl sm:text-5xl font-bold">
              💰 Simple Pricing
            </h2>
            <p className="text-xl text-muted-foreground">
              Choose the plan that works for you
            </p>
          </div>
          <div className="grid sm:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-background p-4 sm:p-8 rounded-2xl shadow-lg border border-primary/10 hover:shadow-xl transition-all duration-300">
              <div className="space-y-4">
                <h3 className="text-2xl font-bold">Basic</h3>
                <p className="text-4xl font-bold text-primary">
                  $9<span className="text-lg text-muted-foreground">/mo</span>
                </p>
                <ul className="space-y-3 text-left">
                  <li className="flex items-center gap-2">
                    <span className="text-green-500">✓</span>
                    <span>Feature 1</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-green-500">✓</span>
                    <span>Feature 2</span>
                  </li>
                </ul>
                <div className="pt-4">
                  <Link
                    href={siteConfig.auth.loginUrl}
                    className="btn btn-outline w-full"
                  >
                    Get Started
                  </Link>
                </div>
              </div>
            </div>
            <div className="bg-background p-4 sm:p-8 rounded-2xl shadow-lg border border-primary/10 hover:shadow-xl transition-all duration-300 relative">
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <span className="bg-primary text-primary-foreground px-3 py-1 rounded-full text-sm font-medium">
                  Popular
                </span>
              </div>
              <div className="space-y-4">
                <h3 className="text-2xl font-bold">Pro</h3>
                <p className="text-4xl font-bold text-primary">
                  $19<span className="text-lg text-muted-foreground">/mo</span>
                </p>
                <ul className="space-y-3 text-left">
                  <li className="flex items-center gap-2">
                    <span className="text-green-500">✓</span>
                    <span>Everything in Basic</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-green-500">✓</span>
                    <span>Pro features</span>
                  </li>
                </ul>
                <div className="pt-4">
                  <Link
                    href={siteConfig.auth.loginUrl}
                    className="btn btn-primary w-full"
                  >
                    Get Started
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section - Band 5 (White Background) */}
      <section id="faq" className="w-full bg-background py-20 px-4 sm:px-8">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-4">
            <h2 className="text-4xl sm:text-5xl font-bold">🤔 FAQ</h2>
            <p className="text-xl text-muted-foreground">
              Quick answers to common questions
            </p>
          </div>
          <div className="space-y-8">
            {["Question 1?", "Question 2?", "Question 3?"].map((q, i) => (
              <div
                key={i}
                className="bg-muted/30 p-4 sm:p-8 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-muted"
              >
                <h3 className="text-xl font-semibold mb-4">{q}</h3>
                <p className="text-lg text-muted-foreground">
                  Answer goes here. Provide helpful information that addresses
                  common concerns or questions your users might have.
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section - Band 6 (Primary Background) */}
      <section className="w-full bg-primary text-primary-foreground py-20 px-4 sm:px-8">
        <div className="max-w-3xl mx-auto text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-4xl sm:text-6xl font-bold">
              Ready to Get Started?
            </h2>
            <p className="text-xl sm:text-2xl opacity-90 max-w-2xl mx-auto">
              Join thousands of happy users today and transform your workflow.
            </p>
          </div>
          <div className="pt-4">
            <Link
              href={siteConfig.auth.loginUrl}
              className="btn btn-accent bg-white text-primary hover:bg-white/90 btn-lg shadow-2xl transform hover:scale-105 transition-all duration-300 text-lg"
            >
              🚀 Sign Up Now
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
