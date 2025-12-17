import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { FileImage, Eye, Brain, MapPin, Percent, FileText } from "lucide-react";
import { format } from "date-fns";

// interface Scan {
//   id: string;
//   scan_date: string;
//   image_url: string;
//   analysis_status: string;
//   tumor_detected: boolean | null;
//   tumor_type: string | null;
//   tumor_location: string | null;
//   confidence_score: number | null;
//   analysis_notes: string | null;
//   created_at: string;
//   patients: {
//     name: string;
//   } | null;
// }

interface Scan {
  id: string;
  scan_date: string;
  image_url: string; // storage path
  signed_image_url?: string; // ✅ ADD THIS
  analysis_status: string;
  tumor_detected: boolean | null;
  tumor_type: string | null;
  tumor_location: string | null;
  confidence_score: number | null;
  analysis_notes: string | null;
  created_at: string;
  patients: {
    name: string;
  } | null;
}


export const ScansTab = () => {
  const [scans, setScans] = useState<Scan[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedScan, setSelectedScan] = useState<Scan | null>(null);

  // useEffect(() => {
  //   const fetchScans = async () => {
  //     const { data, error } = await supabase
  //       .from("mri_scans")
  //       .select(`
  //         *,
  //         patients (name)
  //       `)
  //       .order("created_at", { ascending: false });

  //     if (!error && data) {
  //       setScans(data as Scan[]);
  //     }
  //     setLoading(false);
  //   };

  //   fetchScans();
  // }, []);

useEffect(() => {
  const fetchScans = async () => {
    const { data, error } = await supabase
      .from("mri_scans")
      .select(`*, patients (name)`)
      .order("created_at", { ascending: false });

    if (error || !data) {
      setLoading(false);
      return;
    }

    // 🔑 Create signed URLs
    const scansWithSignedUrls = await Promise.all(
      data.map(async (scan) => {
        const { data: signed, error: signError } =
          await supabase.storage
            .from("mri-scans")
            .createSignedUrl(scan.image_url, 60 * 10); // 10 min

        return {
          ...scan,
          signed_image_url: signError ? "" : signed?.signedUrl,
        };
      })
    );

    setScans(scansWithSignedUrls as Scan[]);
    setLoading(false);
  };

  fetchScans();
}, []);


  const getStatusBadge = (status: string) => {
    const variants: Record<string, { variant: "default" | "secondary" | "destructive" | "outline"; label: string }> = {
      pending: { variant: "secondary", label: "Pending" },
      processing: { variant: "outline", label: "Processing" },
      completed: { variant: "default", label: "Completed" },
      failed: { variant: "destructive", label: "Failed" },
    };
    const config = variants[status] || variants.pending;
    return <Badge variant={config.variant}>{config.label}</Badge>;
  };

  const getResultBadge = (scan: Scan) => {
    if (scan.analysis_status !== "completed" || scan.tumor_detected === null) return null;
    
    return scan.tumor_detected ? (
      <Badge variant="destructive" className="gap-1">
        <Brain className="w-3 h-3" />
        Tumor Detected
      </Badge>
    ) : (
      <Badge className="bg-success hover:bg-success/90 gap-1">
        <Brain className="w-3 h-3" />
        Clear
      </Badge>
    );
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground">Scan History</h2>
        <p className="text-muted-foreground mt-1">
          View all MRI scans and their analysis results
        </p>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6">
                <div className="h-40 bg-muted rounded-lg mb-4" />
                <div className="h-5 bg-muted rounded w-3/4 mb-2" />
                <div className="h-4 bg-muted rounded w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : scans.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <div className="p-4 rounded-full bg-secondary mb-4">
              <FileImage className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-lg font-medium text-foreground">No scans yet</p>
            <p className="text-muted-foreground mt-1">Upload your first MRI scan to see results here</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {scans.map((scan) => (
            <Card
              key={scan.id}
              className="border-border/50 hover:shadow-lg transition-all duration-300 overflow-hidden"
            >
              <div className="aspect-video bg-gradient-to-br from-secondary to-muted relative overflow-hidden">
                <img
                  src={scan.signed_image_url}
                  alt="MRI Scan"
                  className="w-full h-full object-cover opacity-80"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='18' height='18' x='3' y='3' rx='2' ry='2'/%3E%3Ccircle cx='9' cy='9' r='2'/%3E%3Cpath d='m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21'/%3E%3C/svg%3E";
                    (e.target as HTMLImageElement).className = "w-full h-full object-contain p-8 opacity-30";
                  }}
                />
                <div className="absolute top-2 right-2 flex gap-2">
                  {getStatusBadge(scan.analysis_status)}
                </div>
              </div>

              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-foreground truncate">
                    {scan.patients?.name || "Unknown Patient"}
                  </h3>
                  {getResultBadge(scan)}
                </div>

                <p className="text-sm text-muted-foreground mb-4">
                  {format(new Date(scan.scan_date), "MMM dd, yyyy 'at' HH:mm")}
                </p>

                {scan.confidence_score && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
                    <Percent className="w-4 h-4" />
                    <span>Confidence: {scan.confidence_score.toFixed(1)}%</span>
                  </div>
                )}

                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => setSelectedScan(scan)}
                  disabled={scan.analysis_status !== "completed"}
                >
                  <Eye className="w-4 h-4 mr-2" />
                  View Details
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Detail Modal */}
      <Dialog open={!!selectedScan} onOpenChange={() => setSelectedScan(null)}>
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="font-display flex items-center gap-2">
              <Brain className="w-5 h-5 text-accent" />
              Scan Analysis Report
            </DialogTitle>
          </DialogHeader>

          {selectedScan && (
            <div className="space-y-6 mt-4">
              <div className="aspect-video bg-muted rounded-lg overflow-hidden">
                <img
                  src={selectedScan?.signed_image_url}
                  alt="MRI Scan"
                  className="w-full h-full object-contain"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-secondary/30">
                  <CardContent className="p-4">
                    <p className="text-sm text-muted-foreground">Patient</p>
                    <p className="font-semibold">{selectedScan.patients?.name || "Unknown"}</p>
                  </CardContent>
                </Card>

                <Card className="bg-secondary/30">
                  <CardContent className="p-4">
                    <p className="text-sm text-muted-foreground">Scan Date</p>
                    <p className="font-semibold">
                      {format(new Date(selectedScan.scan_date), "MMM dd, yyyy")}
                    </p>
                  </CardContent>
                </Card>
              </div>

              <Card className={selectedScan.tumor_detected ? "bg-destructive/10 border-destructive/30" : "bg-success/10 border-success/30"}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Detection Result
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-2">
                    <strong>Status:</strong>
                    {selectedScan.tumor_detected ? (
                      <Badge variant="destructive">Tumor Detected</Badge>
                    ) : (
                      <Badge className="bg-success hover:bg-success/90">No Tumor Detected</Badge>
                    )}
                  </div>

                  {selectedScan.confidence_score && (
                    <div className="flex items-center gap-2">
                      <Percent className="w-4 h-4 text-muted-foreground" />
                      <span>
                        <strong>Confidence:</strong> {selectedScan.confidence_score.toFixed(1)}%
                      </span>
                    </div>
                  )}

                  {selectedScan.tumor_type && (
                    <div className="flex items-center gap-2">
                      <Brain className="w-4 h-4 text-muted-foreground" />
                      <span>
                        <strong>Type:</strong> {selectedScan.tumor_type}
                      </span>
                    </div>
                  )}

                  {selectedScan.tumor_location && (
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-muted-foreground" />
                      <span>
                        <strong>Location:</strong> {selectedScan.tumor_location}
                      </span>
                    </div>
                  )}
                </CardContent>
              </Card>

              {selectedScan.analysis_notes && (
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      Analysis Notes
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{selectedScan.analysis_notes}</p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};
