import { requireAuth } from "@/lib/auth/server";

export default async function DashboardPage() {
  const user = await requireAuth();

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="max-w-2xl">
        <h1 className="text-2xl md:text-4xl font-bold mb-6">
          Welcome to your Dashboard
        </h1>

        <div className="card bg-base-200 shadow-xl mb-6">
          <div className="card-body">
            <h2 className="card-title">Your Account</h2>
            <div className="space-y-2">
              <p>
                <span className="font-semibold">Email:</span> {user.email}
              </p>
              <p>
                <span className="font-semibold">User ID:</span> {user.id}
              </p>
              <p>
                <span className="font-semibold">Name:</span> {user.name}
              </p>
              {user.image && (
                <p>
                  <span className="font-semibold">Profile Image:</span>{" "}
                  <img
                    src={user.image}
                    alt="Profile"
                    className="w-8 h-8 rounded-full inline-block ml-2"
                  />
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
