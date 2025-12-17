import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Brain, Mail, Lock, User, ArrowRight, Loader2 } from "lucide-react";

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) {
        navigate("/dashboard");
      }
    });

    // Also check current session on mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        navigate("/dashboard");
      }
    });

    const subscription = data?.subscription;
    return () => {
      if (subscription && typeof subscription.unsubscribe === "function") {
        subscription.unsubscribe();
      }
    };
  }, [navigate]);

  // const handleAuth = async (e: React.FormEvent) => {
  //   e.preventDefault();
  //   setLoading(true);

  //   try {
  //     if (isLogin) {
  //       const { error } = await supabase.auth.signInWithPassword({
  //         email,
  //         password,
  //       });
  //       if (error) throw error;
  //       toast.success("Welcome back, Doctor!");
  //     } else {
  //       const { error } = await supabase.auth.signUp({
  //         email,
  //         password,
  //         options: {
  //           emailRedirectTo: `${window.location.origin}/dashboard`,
  //           data: {
  //             full_name: fullName,
  //           },
  //         },
  //       });
  //       if (error) throw error;
  //       toast.success("Account created successfully!");
  //     }
  //   } catch (error: any) {
  //     if (error.message.includes("User already registered")) {
  //       toast.error("This email is already registered. Please sign in.");
  //     } else {
  //       toast.error(error.message);
  //     }
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        // Login flow
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) throw error;

        // If a session is returned, user is signed in immediately
        if (data?.session) {
          toast.success(
            `Welcome back, Dr. ${data.user?.user_metadata?.full_name ?? ""}`
          );
          navigate("/dashboard");
        } else {
          // No session (e.g. magic link or needs email confirmation)
          toast.success("Check your email to complete sign in.");
        }
      } else {
        // Signup flow
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${window.location.origin}/dashboard`,
            data: {
              full_name: fullName,
            },
          },
        });

        if (error) {
          // Common duplicate account error handling
          const msg = (error as any)?.message ?? "";
          const status = (error as any)?.status;
          if (status === 400 && /registered|already exists|already registered/i.test(msg)) {
            toast.error("This email is already registered. Please sign in.");
            return;
          }
          throw error;
        }

        // If session exists, user is signed in immediately
        if (data?.session) {
          toast.success("Account created and signed in!");
          navigate("/dashboard");
        } else {
          // Otherwise, instruct user to confirm their email
          toast.success(
            "Account created. Check your email to confirm and complete sign in."
          );
        }
      }
    } catch (err: any) {
      const message = err?.message ?? String(err);
      if (/already registered|already exists/i.test(message)) {
        toast.error("This email is already registered. Please sign in.");
      } else {
        toast.error(message);
      }
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden" style={{ background: 'linear-gradient(135deg, hsl(215, 70%, 25%) 0%, hsl(200, 80%, 35%) 50%, hsl(175, 65%, 40%) 100%)' }}>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%23ffffff%22%20fill-opacity%3D%220.05%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%222%22%2F%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E')]" />
        
        <div className="relative z-10 flex flex-col justify-center items-center w-full p-12 text-primary-foreground">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-4 rounded-2xl bg-primary-foreground/10 backdrop-blur-sm">
              <Brain className="w-12 h-12" />
            </div>
          </div>
          
          <h1 className="text-4xl font-display font-bold mb-4 text-center">
            NeuroScan AI
          </h1>
          <p className="text-xl text-primary-foreground/80 text-center max-w-md mb-8">
            Advanced Brain Tumor Detection Platform for Medical Professionals
          </p>
          
          <div className="grid grid-cols-2 gap-6 mt-8">
            {[
              { label: "Accuracy Rate", value: "98.5%" },
              { label: "Scans Analyzed", value: "50K+" },
              { label: "Detection Time", value: "<2 min" },
              { label: "Doctors Trust", value: "1,200+" },
            ].map((stat) => (
              <div key={stat.label} className="text-center p-4 rounded-xl bg-primary-foreground/10 backdrop-blur-sm">
                <div className="text-2xl font-bold">{stat.value}</div>
                <div className="text-sm text-primary-foreground/70">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right side - Auth Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-background">
        <div className="w-full max-w-md space-y-8">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center justify-center gap-3 mb-8">
            <div className="p-3 rounded-xl bg-primary text-primary-foreground">
              <Brain className="w-8 h-8" />
            </div>
            <span className="text-2xl font-display font-bold text-foreground">NeuroScan AI</span>
          </div>

          <div className="text-center lg:text-left">
            <h2 className="text-3xl font-display font-bold text-foreground">
              {isLogin ? "Welcome back" : "Create account"}
            </h2>
            <p className="mt-2 text-muted-foreground">
              {isLogin
                ? "Sign in to access your dashboard"
                : "Register to start analyzing MRI scans"}
            </p>
          </div>

          <form onSubmit={handleAuth} className="space-y-6">
            {!isLogin && (
              <div className="space-y-2">
                <Label htmlFor="fullName" className="text-foreground">Full Name</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    id="fullName"
                    type="text"
                    placeholder="Dr. John Smith"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="pl-10 h-12 bg-secondary border-border"
                    required={!isLogin}
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email" className="text-foreground">Email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="doctor@hospital.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="pl-10 h-12 bg-secondary border-border"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-foreground">Password</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 h-12 bg-secondary border-border"
                  required
                  minLength={6}
                />
              </div>
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full h-12 text-base font-semibold"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  {isLogin ? "Sign In" : "Create Account"}
                  <ArrowRight className="ml-2 w-5 h-5" />
                </>
              )}
            </Button>
          </form>

          <div className="text-center">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-accent hover:text-accent/80 font-medium transition-colors"
            >
              {isLogin
                ? "Don't have an account? Sign up"
                : "Already have an account? Sign in"}
            </button>
          </div>

          <p className="text-xs text-center text-muted-foreground">
            By continuing, you agree to our Terms of Service and Privacy Policy.
            This platform is for authorized medical professionals only.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Auth;
