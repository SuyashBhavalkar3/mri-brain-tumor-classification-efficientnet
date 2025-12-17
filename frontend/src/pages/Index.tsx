import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Brain, Shield, Zap, ChartLine, ArrowRight } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        navigate("/dashboard");
      }
    });
  }, [navigate]);

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Detection",
      description: "Advanced deep learning models analyze MRI scans with 98.5% accuracy",
    },
    {
      icon: Zap,
      title: "Rapid Analysis",
      description: "Get comprehensive results in under 2 minutes",
    },
    {
      icon: Shield,
      title: "HIPAA Compliant",
      description: "Enterprise-grade security for patient data protection",
    },
    {
      icon: ChartLine,
      title: "Detailed Reports",
      description: "Comprehensive visualization and analysis reports",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0" style={{ background: 'linear-gradient(135deg, hsl(215, 70%, 25%) 0%, hsl(200, 80%, 35%) 50%, hsl(175, 65%, 40%) 100%)' }}>
          <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%23ffffff%22%20fill-opacity%3D%220.03%22%3E%3Ccircle%20cx%3D%2230%22%20cy%3D%2230%22%20r%3D%222%22%2F%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E')]" />
        </div>

        <nav className="relative z-10 container mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-primary-foreground/10 backdrop-blur-sm">
              <Brain className="w-8 h-8 text-primary-foreground" />
            </div>
            <span className="text-xl font-display font-bold text-primary-foreground">NeuroScan AI</span>
          </div>
          <Button
            onClick={() => navigate("/auth")}
            variant="secondary"
            className="gap-2"
          >
            Doctor Login
            <ArrowRight className="w-4 h-4" />
          </Button>
        </nav>

        <div className="relative z-10 container mx-auto px-6 py-24 text-center">
          <h1 className="text-4xl md:text-6xl font-display font-bold text-primary-foreground mb-6 leading-tight">
            Advanced Brain Tumor
            <br />
            Detection Platform
          </h1>
          <p className="text-xl text-primary-foreground/80 max-w-2xl mx-auto mb-10">
            Harness the power of artificial intelligence for accurate and rapid MRI scan analysis.
            Trusted by over 1,200 medical professionals worldwide.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              size="lg"
              onClick={() => navigate("/auth")}
              className="bg-primary-foreground text-primary hover:bg-primary-foreground/90 gap-2 text-lg h-14 px-8"
            >
              Get Started
              <ArrowRight className="w-5 h-5" />
            </Button>
          <Button
              size="lg"
              onClick={() =>
                window.open(
                  "https://my.clevelandclinic.org/health/diseases/6149-brain-cancer-brain-tumor",
                  "_blank"
                )
              }
              variant="outline"
              className="border-primary-foreground text-primary hover:bg-primary-foreground/10 text-lg h-14 px-8"
            >
              Learn More
          </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-20 max-w-4xl mx-auto">
            {[
              { value: "98.5%", label: "Accuracy Rate" },
              { value: "50K+", label: "Scans Analyzed" },
              { value: "<2 min", label: "Analysis Time" },
              { value: "1,200+", label: "Active Doctors" },
            ].map((stat) => (
              <div key={stat.label} className="p-4 rounded-xl bg-primary-foreground/10 backdrop-blur-sm">
                <div className="text-3xl font-display font-bold text-primary-foreground">{stat.value}</div>
                <div className="text-sm text-primary-foreground/70 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Wave decoration */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M0 120L60 110C120 100 240 80 360 70C480 60 600 60 720 65C840 70 960 80 1080 85C1200 90 1320 90 1380 90L1440 90V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z"
              fill="hsl(var(--background))"
            />
          </svg>
        </div>
      </header>

      {/* Features Section */}
      <section className="py-24 container mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
            Why Choose NeuroScan AI?
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Our platform combines cutting-edge AI technology with intuitive design
            to deliver the most accurate brain tumor detection available.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="p-6 rounded-2xl bg-card border border-border/50 hover:shadow-lg hover:border-accent/30 transition-all duration-300"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="p-3 rounded-xl bg-accent/10 w-fit mb-4">
                <feature.icon className="w-6 h-6 text-accent" />
              </div>
              <h3 className="text-lg font-display font-semibold text-foreground mb-2">
                {feature.title}
              </h3>
              <p className="text-muted-foreground">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-secondary/30">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
            Ready to Transform Your Practice?
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-8">
            Join thousands of medical professionals using AI-powered diagnostics
            to improve patient outcomes.
          </p>
          <Button
            size="lg"
            onClick={() => navigate("/auth")}
            className="gap-2 text-lg h-14 px-8"
          >
            Start Free Trial
            <ArrowRight className="w-5 h-5" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-primary/10">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <span className="font-display font-semibold text-foreground">NeuroScan AI</span>
            </div>
            <p className="text-sm text-muted-foreground">
              © 2024 NeuroScan AI. For authorized medical professionals only.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
