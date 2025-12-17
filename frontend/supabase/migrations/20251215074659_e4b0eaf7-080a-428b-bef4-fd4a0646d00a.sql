-- =====================================================
-- EXTENSIONS
-- =====================================================
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =====================================================
-- PROFILES (DOCTORS)
-- =====================================================
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT NOT NULL,
  specialization TEXT DEFAULT 'Neurology',
  license_number TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Profiles: select own"
  ON public.profiles
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Profiles: insert own"
  ON public.profiles
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Profiles: update own"
  ON public.profiles
  FOR UPDATE
  USING (auth.uid() = user_id);

-- =====================================================
-- PATIENTS
-- =====================================================
CREATE TABLE public.patients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doctor_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  date_of_birth DATE,
  gender TEXT CHECK (gender IN ('male', 'female', 'other')),
  contact_email TEXT,
  contact_phone TEXT,
  medical_history TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Patients: select own"
  ON public.patients
  FOR SELECT
  USING (auth.uid() = doctor_id);

CREATE POLICY "Patients: insert own"
  ON public.patients
  FOR INSERT
  WITH CHECK (auth.uid() = doctor_id);

CREATE POLICY "Patients: update own"
  ON public.patients
  FOR UPDATE
  USING (auth.uid() = doctor_id);

CREATE POLICY "Patients: delete own"
  ON public.patients
  FOR DELETE
  USING (auth.uid() = doctor_id);

-- =====================================================
-- MRI SCANS
-- =====================================================
CREATE TABLE public.mri_scans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  doctor_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  scan_date TIMESTAMPTZ NOT NULL DEFAULT now(),
  image_url TEXT NOT NULL,
  analysis_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (analysis_status IN ('pending', 'processing', 'completed', 'failed')),
  tumor_detected BOOLEAN,
  tumor_type TEXT,
  tumor_location TEXT,
  confidence_score DECIMAL(5,2) CHECK (confidence_score BETWEEN 0 AND 100),
  analysis_notes TEXT,
  report_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.mri_scans ENABLE ROW LEVEL SECURITY;

-- SELECT: only scans of own patients
CREATE POLICY "MRI: select own patients"
  ON public.mri_scans
  FOR SELECT
  USING (
    auth.uid() = doctor_id
    AND EXISTS (
      SELECT 1
      FROM public.patients p
      WHERE p.id = patient_id
        AND p.doctor_id = auth.uid()
    )
  );

-- INSERT: only for own patients
CREATE POLICY "MRI: insert own patients"
  ON public.mri_scans
  FOR INSERT
  WITH CHECK (
    auth.uid() = doctor_id
    AND EXISTS (
      SELECT 1
      FROM public.patients p
      WHERE p.id = patient_id
        AND p.doctor_id = auth.uid()
    )
  );

CREATE POLICY "MRI: update own"
  ON public.mri_scans
  FOR UPDATE
  USING (auth.uid() = doctor_id);

CREATE POLICY "MRI: delete own"
  ON public.mri_scans
  FOR DELETE
  USING (auth.uid() = doctor_id);

-- =====================================================
-- UPDATED_AT TRIGGER
-- =====================================================
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER patients_updated_at
  BEFORE UPDATE ON public.patients
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER mri_scans_updated_at
  BEFORE UPDATE ON public.mri_scans
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- =====================================================
-- AUTO CREATE PROFILE ON SIGNUP
-- =====================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id, full_name)
  VALUES (
    NEW.id,
    COALESCE(NEW.raw_user_meta_data ->> 'full_name', 'Doctor')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =====================================================
-- STORAGE (MRI SCANS)
-- =====================================================
INSERT INTO storage.buckets (id, name, public)
VALUES ('mri-scans', 'mri-scans', false);

CREATE POLICY "Storage: upload MRI"
  ON storage.objects
  FOR INSERT
  WITH CHECK (
    bucket_id = 'mri-scans'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Storage: view MRI"
  ON storage.objects
  FOR SELECT
  USING (
    bucket_id = 'mri-scans'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Storage: delete MRI"
  ON storage.objects
  FOR DELETE
  USING (
    bucket_id = 'mri-scans'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );