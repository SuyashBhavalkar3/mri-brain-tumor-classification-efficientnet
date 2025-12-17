import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Upload, FileImage, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface Patient {
  id: string;
  name: string;
}

export const UploadTab = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    const fetchPatients = async () => {
      const { data } = await supabase.from("patients").select("id, name").order("name");
      if (data) setPatients(data);
    };
    fetchPatients();
  }, []);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith("image/")) {
        setFile(droppedFile);
      } else {
        toast.error("Please upload an image file");
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  // const handleUpload = async () => {
  //   if (!file || !selectedPatient) {
  //     toast.error("Please select a patient and upload a file");
  //     return;
  //   }

  //   setUploading(true);

  //   try {
  //     const { data: { user } } = await supabase.auth.getUser();
  //     if (!user) throw new Error("Not authenticated");

  //     // Upload file to storage
  //     const fileExt = file.name.split(".").pop();
  //     const fileName = `${user.id}/${Date.now()}.${fileExt}`;
      
  //     const { error: uploadError } = await supabase.storage
  //       .from("mri-scans")
  //       .upload(fileName, file);

  //     if (uploadError) throw uploadError;

  //     // Get public URL
  //     const { data: { publicUrl } } = supabase.storage
  //       .from("mri-scans")
  //       .getPublicUrl(fileName);

  //     // Create scan record
  //     const { error: insertError } = await supabase.from("mri_scans").insert({
  //       patient_id: selectedPatient,
  //       doctor_id: user.id,
  //       image_url: publicUrl,
  //       analysis_status: "pending",
  //     });

  //     if (insertError) throw insertError;

  //     toast.success("MRI scan uploaded successfully! Analysis will begin shortly.");
  //     setFile(null);
  //     setSelectedPatient("");

  //     // Simulate analysis (in real app, this would trigger AI analysis)
  //     setTimeout(async () => {
  //       // Mock analysis result
  //       const tumorDetected = Math.random() > 0.7;
  //       await supabase
  //         .from("mri_scans")
  //         .update({
  //           analysis_status: "completed",
  //           tumor_detected: tumorDetected,
  //           confidence_score: 85 + Math.random() * 14,
  //           tumor_type: tumorDetected ? "Glioblastoma" : null,
  //           tumor_location: tumorDetected ? "Frontal Lobe" : null,
  //           analysis_notes: tumorDetected
  //             ? "Abnormal mass detected in the frontal lobe region. Recommend further investigation."
  //             : "No abnormalities detected. Brain tissue appears normal.",
  //         })
  //         .eq("image_url", publicUrl);

  //       toast.info("Analysis completed! Check the scan history for results.");
  //     }, 3000);

  //   } catch (error: any) {
  //     toast.error(error.message || "Failed to upload scan");
  //   } finally {
  //     setUploading(false);
  //   }
  // };

const handleUpload = async () => {
  if (!file || !selectedPatient) {
    toast.error("Please select a patient and upload a file");
    return;
  }

  setUploading(true);

  try {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) throw new Error("Not authenticated");

    // Step 1: Upload file to Supabase (optional, you can also send directly to backend)
    const fileExt = file.name.split(".").pop();
    const fileName = `${user.id}/${Date.now()}.${fileExt}`;
    
    const { error: uploadError } = await supabase.storage
      .from("mri-scans")
      .upload(fileName, file);

    if (uploadError) throw uploadError;

    const { data: { publicUrl } } = supabase.storage
      .from("mri-scans")
      .getPublicUrl(fileName);

    // Step 2: Send file to backend for analysis
    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", selectedPatient);
    formData.append("doctor_id", user.id);

    const response = await fetch("https://mri-brain-tumor-classification.onrender.com/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Failed to analyze scan");

    const analysisResult = await response.json();

    // Step 3: Save analysis result in Supabase
    // const { error: insertError } = await supabase.from("mri_scans").insert({
    //   patient_id: selectedPatient,
    //   doctor_id: user.id,
    //   image_url: fileName,
    //   analysis_status: "completed",
    //   tumor_detected: analysisResult.tumor_detected,
    //   confidence_score: analysisResult.confidence_score,
    //   tumor_type: analysisResult.tumor_type,
    //   tumor_location: analysisResult.tumor_location,
    //   analysis_notes: analysisResult.analysis_notes,
    // });

    const { error: insertError } = await supabase.from("mri_scans").insert({
    patient_id: selectedPatient,
    doctor_id: user.id,
    image_url: fileName, // 🔑 STORAGE PATH ONLY
    analysis_status: "completed",
    tumor_detected: analysisResult.tumor_detected,
    confidence_score: analysisResult.confidence_score,
    tumor_type: analysisResult.tumor_type,
    tumor_location: analysisResult.tumor_location,
    analysis_notes: analysisResult.analysis_notes,
  });


    if (insertError) throw insertError;

    toast.success("MRI scan uploaded and analyzed successfully!");

    setFile(null);
    setSelectedPatient("");

  } catch (error: any) {
    toast.error(error.message || "Failed to upload and analyze scan");
  } finally {
    setUploading(false);
  }
};


  return (
    <div className="space-y-6 animate-fade-in max-w-3xl">
      <div>
        <h2 className="text-2xl font-display font-bold text-foreground">Upload MRI Scan</h2>
        <p className="text-muted-foreground mt-1">
          Upload a patient's MRI scan for AI-powered tumor detection analysis
        </p>
      </div>

      <Card className="border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 font-display">
            <FileImage className="w-5 h-5 text-accent" />
            New Scan Upload
          </CardTitle>
          <CardDescription>
            Select a patient and upload their MRI scan image
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Patient Selection */}
          <div className="space-y-2">
            <Label htmlFor="patient">Select Patient</Label>
            {/* <Select value={selectedPatient} onValueChange={setSelectedPatient}>
              <SelectTrigger id="patient" className="bg-secondary/50">
                <SelectValue placeholder="Choose a patient" />
              </SelectTrigger>
              <SelectContent>
                {patients.length === 0 ? (
                  <SelectItem value="" disabled>
                    No patients found. Add a patient first.
                  </SelectItem>
                ) : (
                  patients.map((patient) => (
                    <SelectItem key={patient.id} value={patient.id}>
                      {patient.name}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select> */}
            <Select value={selectedPatient} onValueChange={setSelectedPatient}>
  <SelectTrigger id="patient" className="bg-secondary/50">
    <SelectValue placeholder="Choose a patient" />
  </SelectTrigger>
  <SelectContent>
    {patients.length === 0 ? (
      <SelectItem value="no-patient" disabled>
        No patients found. Add a patient first.
      </SelectItem>
    ) : (
      patients.map((patient) => (
        <SelectItem key={patient.id} value={patient.id}>
          {patient.name}
        </SelectItem>
      ))
    )}
  </SelectContent>
</Select>

          </div>

          {/* File Upload Zone */}
          <div className="space-y-2">
            <Label>MRI Scan Image</Label>
            <div
              className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
                dragActive
                  ? "border-accent bg-accent/5"
                  : file
                  ? "border-success bg-success/5"
                  : "border-border hover:border-accent/50 hover:bg-secondary/30"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              
              {file ? (
                <div className="flex flex-col items-center gap-3">
                  <CheckCircle className="w-12 h-12 text-success" />
                  <div>
                    <p className="font-medium text-foreground">{file.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                    }}
                  >
                    Remove
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <div className="p-4 rounded-full bg-secondary">
                    <Upload className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <div>
                    <p className="font-medium text-foreground">
                      Drop your MRI scan here or click to browse
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Supports JPEG, PNG, DICOM formats up to 50MB
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Info Alert */}
          <div className="flex items-start gap-3 p-4 rounded-lg bg-accent/10 border border-accent/20">
            <AlertCircle className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-foreground">AI Analysis Notice</p>
              <p className="text-muted-foreground mt-1">
                Our AI model will analyze the scan and provide detection results within minutes.
                Results should be verified by a medical professional.
              </p>
            </div>
          </div>

          {/* Submit Button */}
          <Button
            onClick={handleUpload}
            disabled={!file || !selectedPatient || uploading}
            className="w-full h-12"
          >
            {uploading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Uploading & Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-5 h-5 mr-2" />
                Upload & Analyze Scan
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};
