import { useEffect, useState, useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import {
  User,
  CalendarClock,
  AlertTriangle,
  CheckCircle,
  Hourglass,
  Tag,
  Percent,
} from "lucide-react";

interface ScanRecord {
  id: string;
  created_at: string; // upload timestamp
  scan_date: string; // original scan date (may be null)
  analysis_status: string;
  tumor_detected: boolean | null;
  confidence_score: number | null;
  tumor_type: string | null;
  patients: { id: string; name: string } | null;
}

type GroupedScans = Record<string, ScanRecord[]>;

export const RecentScansTable = () => {
  const [scans, setScans] = useState<ScanRecord[] | null>(null);

  useEffect(() => {
    const fetchScans = async () => {
      const { data, error } = await supabase
        .from("mri_scans")
        .select(`
          id,
          created_at,
          scan_date,
          analysis_status,
          tumor_detected,
          confidence_score,
          tumor_type,
          patients(id, name)
        `)
        .order("created_at", { ascending: false });

      if (!error && data) {
        setScans(data as ScanRecord[]);
      } else {
        setScans([]);
      }
    };

    fetchScans();
  }, []);

  const loading = scans === null;

  const groupedByPatient: GroupedScans = useMemo(() => {
    if (!scans || scans.length === 0) return {};
    return scans.reduce((acc: GroupedScans, s) => {
      const name = s.patients?.name || "Unknown Patient";
      if (!acc[name]) acc[name] = [];
      acc[name].push(s);
      return acc;
    }, {});
  }, [scans]);

  const formatConfidence = (val: number | null) => (val === null || val === undefined ? "N/A" : `${val.toFixed(2)}%`);

  const analysisBadge = (status: string) => {
    const key = status?.toLowerCase() || "pending";
    if (key === "completed") return <Badge variant="default">Completed</Badge>;
    if (key === "processing") return <Badge variant="outline">Processing</Badge>;
    return <Badge variant="secondary">Pending</Badge>;
  };

  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-card/40">
            <div className="w-10 h-10 rounded-md bg-muted animate-pulse" />
            <div className="flex-1">
              <div className="h-4 bg-muted animate-pulse rounded w-3/5 mb-2" />
              <div className="h-3 bg-muted animate-pulse rounded w-2/5" />
            </div>
            <div className="w-24 h-6 bg-muted animate-pulse rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (!scans || scans.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <div className="flex items-center justify-center gap-2">
          <FileIconPlaceholder />
        </div>
        <p className="mt-3">No scan history available. Scans will appear here after upload and analysis.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {Object.keys(groupedByPatient).map((patientName) => (
        <div key={patientName}>
          <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
            <User className="w-4 h-4 text-muted-foreground" /> {patientName}
          </h3>
          <div className="mt-2 space-y-2">
            {groupedByPatient[patientName]
              .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
              .map((scan) => (
                <div
                  key={scan.id}
                  className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 p-3 rounded-lg bg-card hover:shadow-sm transition-all"
                >
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    <div className="p-2 rounded-md bg-accent/10">
                      <CalendarClock className="w-5 h-5 text-accent" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium truncate">{format(new Date(scan.created_at), "MMM dd, yyyy, hh:mm a")}</p>
                      <p className="text-xs text-muted-foreground truncate">
                        Scan date: {scan.scan_date ? format(new Date(scan.scan_date), "MMM dd, yyyy") : "N/A"}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 flex-wrap">
                    <div className="flex items-center gap-2">
                      {scan.analysis_status === "completed" && scan.tumor_detected !== null ? (
                        scan.tumor_detected ? (
                          <Badge variant="destructive" className="flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" /> Tumor Detected
                          </Badge>
                        ) : (
                          <Badge variant="default" className="flex items-center gap-2 bg-success text-white">
                            <CheckCircle className="w-4 h-4" /> No Tumor
                          </Badge>
                        )
                      ) : (
                        <Badge variant="outline" className="flex items-center gap-2">
                          <Hourglass className="w-4 h-4" /> Pending
                        </Badge>
                      )}
                    </div>

                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Tag className="w-4 h-4" /> {scan.tumor_type || "N/A"}
                    </div>

                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Percent className="w-4 h-4" /> {formatConfidence(scan.confidence_score)}
                    </div>

                    <div>{analysisBadge(scan.analysis_status)}</div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
};

function FileIconPlaceholder() {
  return (
    <div className="flex items-center gap-2 text-muted-foreground">
      <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
      </svg>
    </div>
  );
}
