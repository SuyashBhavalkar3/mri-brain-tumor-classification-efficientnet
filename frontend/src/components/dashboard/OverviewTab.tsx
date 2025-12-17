import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Users, FileImage, AlertTriangle, TrendingUp, Activity } from "lucide-react";
import { RecentScansTable } from "./RecentScansTable";
import { AnalyticsChart } from "./AnalyticsChart";

interface Stats {
  totalPatients: number;
  totalScans: number;
  tumorsDetected: number;
  pendingAnalysis: number;
}

export const OverviewTab = () => {
  const [stats, setStats] = useState<Stats>({
    totalPatients: 0,
    totalScans: 0,
    tumorsDetected: 0,
    pendingAnalysis: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const [patientsRes, scansRes] = await Promise.all([
        supabase.from("patients").select("id", { count: "exact" }),
        supabase.from("mri_scans").select("*"),
      ]);

      const scans = scansRes.data || [];
      setStats({
        totalPatients: patientsRes.count || 0,
        totalScans: scans.length,
        tumorsDetected: scans.filter((s) => s.tumor_detected === true).length,
        pendingAnalysis: scans.filter((s) => s.analysis_status === "pending" || s.analysis_status === "processing").length,
      });
      setLoading(false);
    };

    fetchStats();
  }, []);

  const statCards = [
    {
      title: "Total Patients",
      value: stats.totalPatients,
      icon: Users,
      color: "bg-primary/10 text-primary",
      trend: "+12%",
    },
    {
      title: "MRI Scans",
      value: stats.totalScans,
      icon: FileImage,
      color: "bg-accent/10 text-accent",
      trend: "+8%",
    },
    {
      title: "Tumors Detected",
      value: stats.tumorsDetected,
      icon: AlertTriangle,
      color: "bg-destructive/10 text-destructive",
      trend: stats.tumorsDetected > 0 ? `${((stats.tumorsDetected / Math.max(stats.totalScans, 1)) * 100).toFixed(1)}%` : "0%",
    },
    {
      title: "Pending Analysis",
      value: stats.pendingAnalysis,
      icon: Activity,
      color: "bg-warning/10 text-warning",
      trend: "Active",
    },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground">Dashboard Overview</h2>
        <p className="text-muted-foreground mt-1">
          Monitor your patients and scan analysis in real-time
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat, index) => (
          <Card
            key={stat.title}
            className="border-border/50 bg-card hover:shadow-lg transition-all duration-300"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className={`p-3 rounded-xl ${stat.color}`}>
                  <stat.icon className="w-6 h-6" />
                </div>
                <div className="flex items-center gap-1 text-xs font-medium text-success">
                  <TrendingUp className="w-3 h-3" />
                  {stat.trend}
                </div>
              </div>
              <div className="mt-4">
                <p className="text-3xl font-display font-bold text-foreground">
                  {loading ? "..." : stat.value}
                </p>
                <p className="text-sm text-muted-foreground mt-1">{stat.title}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts and Recent Scans */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-display">
              <Brain className="w-5 h-5 text-accent" />
              Detection Analytics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <AnalyticsChart />
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 font-display">
              <FileImage className="w-5 h-5 text-accent" />
              Recent Scans
            </CardTitle>
          </CardHeader>
          <CardContent>
            <RecentScansTable />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
