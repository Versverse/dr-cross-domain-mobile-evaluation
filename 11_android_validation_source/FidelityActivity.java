package com.example.aptosedgetest.fidelity20261016;

import android.app.Activity;
import android.os.Bundle;
import android.os.Build;
import android.os.PowerManager;
import android.os.BatteryManager;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ApplicationInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.Button;
import android.widget.TextView;
import android.widget.LinearLayout;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgcodecs.Imgcodecs;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.TensorFlowLite;
import org.json.JSONObject;
import org.json.JSONArray;
import java.io.*;
import java.nio.*;
import java.security.MessageDigest;
import java.util.*;

/** Independent debug experiment; no historical performance measurements. */
public final class FidelityActivity extends Activity {
    private TextView status;
    private Button button;
    private File inputRoot;
    private static final String RECORD = "NEW_PAIRED_FIDELITY_20261016";

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        inputRoot = new File(getFilesDir(), "fidelity_inputs");
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL); layout.setPadding(28,36,28,24);
        status = new TextView(this);
        status.setText("DR paired fidelity 20261016\nNative Java debug experiment\n\nInputs: "+inputRoot+"\n\nTimes are diagnostic only. No historical timing replacement.");
        button = new Button(this); button.setText("Run frozen 218-image experiment");
        button.setOnClickListener(v -> startRun());
        layout.addView(status); layout.addView(button); setContentView(layout);
    }
    private void report(String message) { runOnUiThread(() -> status.setText(message)); }
    private static void require(boolean ok, String message) { if (!ok) throw new IllegalStateException(message); }
    private static byte[] read(InputStream stream) throws IOException {
        try (InputStream in=stream; ByteArrayOutputStream out=new ByteArrayOutputStream()) {
            byte[] b=new byte[1024*1024]; int n;
            while ((n=in.read(b))!=-1) out.write(b,0,n);
            return out.toByteArray();
        }
    }
    private static byte[] read(File file) throws IOException { return read(new FileInputStream(file)); }
    private static void write(File file, byte[] b) throws IOException {
        File parent=file.getParentFile(); if (parent!=null) parent.mkdirs();
        try (OutputStream out=new FileOutputStream(file)) { out.write(b); }
    }
    private static void write(File f,String s) throws IOException { write(f,s.getBytes("UTF-8")); }
    private static String sha(byte[] bytes) throws Exception {
        byte[] digest=MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder s=new StringBuilder(); for(byte b:digest) s.append(String.format(Locale.US,"%02x",b&255)); return s.toString();
    }
    private static String sha(File file) throws Exception {
        MessageDigest md=MessageDigest.getInstance("SHA-256");
        try(InputStream in=new FileInputStream(file)) { byte[] b=new byte[1048576];int n;while((n=in.read(b))!=-1)md.update(b,0,n); }
        StringBuilder s=new StringBuilder();for(byte b:md.digest())s.append(String.format(Locale.US,"%02x",b&255));return s.toString();
    }
    private File safe(String relative) throws IOException {
        File f=new File(inputRoot,relative).getCanonicalFile();
        require(f.getPath().startsWith(inputRoot.getCanonicalPath()+File.separator)&&f.isFile(),"Invalid input "+relative);return f;
    }
    private static ByteBuffer direct(byte[] bytes) {
        ByteBuffer b=ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.LITTLE_ENDIAN);b.put(bytes);b.rewind();return b;
    }
    private static byte[] bytes(ByteBuffer b) { ByteBuffer c=b.duplicate();c.rewind();byte[] out=new byte[c.capacity()];c.get(out);return out; }
    private static float[] floats(byte[] bytes) {
        require(bytes.length==224*224*3*4,"Tensor size mismatch");FloatBuffer b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();float[] f=new float[b.remaining()];b.get(f);return f;
    }
    private static JSONArray json(double[] values) throws Exception { JSONArray a=new JSONArray();for(double x:values)a.put(x);return a; }
    private JSONObject preflight() throws Exception {
        Intent i=registerReceiver(null,new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        int level=i==null?-1:i.getIntExtra(BatteryManager.EXTRA_LEVEL,-1), scale=i==null?-1:i.getIntExtra(BatteryManager.EXTRA_SCALE,-1);
        int bs=i==null?-1:i.getIntExtra(BatteryManager.EXTRA_STATUS,-1), temp=i==null?Integer.MIN_VALUE:i.getIntExtra(BatteryManager.EXTRA_TEMPERATURE,Integer.MIN_VALUE);
        PowerManager p=(PowerManager)getSystemService(POWER_SERVICE);
        return new JSONObject().put("battery_percent",level>=0&&scale>0?100.0*level/scale:-1).put("battery_status_code",bs)
            .put("charging_or_full",bs==BatteryManager.BATTERY_STATUS_CHARGING||bs==BatteryManager.BATTERY_STATUS_FULL)
            .put("plugged_code",i==null?-1:i.getIntExtra(BatteryManager.EXTRA_PLUGGED,-1)).put("battery_temperature_C",temp==Integer.MIN_VALUE?JSONObject.NULL:temp/10.0)
            .put("power_save_mode",p.isPowerSaveMode()).put("thermal_status",Build.VERSION.SDK_INT>=29?p.getCurrentThermalStatus():JSONObject.NULL)
            .put("debuggable_build",(getApplicationInfo().flags&ApplicationInfo.FLAG_DEBUGGABLE)!=0).put("timing_role","Diagnostic fidelity times only; never a formal performance benchmark.");
    }
    private static Mat rawBgr(File file) {
        Bitmap bitmap=BitmapFactory.decodeFile(file.getPath());require(bitmap!=null,"Bitmap decode failed "+file.getName());
        Mat rgba=new Mat(),bgr=new Mat();Utils.bitmapToMat(bitmap,rgba);bitmap.recycle();Imgproc.cvtColor(rgba,bgr,Imgproc.COLOR_RGBA2BGR);rgba.release();return bgr;
    }
    /** Java translation of DrPipeline.preprocessFundusBgr; integer crop semantics unchanged. */
    private static Mat preprocess(Mat input) {
        Mat gray=new Mat(),threshold=new Mat(),hierarchy=new Mat(),cropped=null;
        List<MatOfPoint> contours=new ArrayList<>();
        try {
            Imgproc.cvtColor(input,gray,Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(gray,threshold,15.0,255.0,Imgproc.THRESH_BINARY);
            Imgproc.findContours(threshold,contours,hierarchy,Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
            if(contours.isEmpty()) cropped=input.clone();
            else {
                MatOfPoint largest=contours.get(0);double area=Imgproc.contourArea(largest);
                for(int k=1;k<contours.size();k++){double a=Imgproc.contourArea(contours.get(k));if(a>area){area=a;largest=contours.get(k);}}
                Rect r=Imgproc.boundingRect(largest);int cx=r.x+r.width/2,cy=r.y+r.height/2,side=Math.max(r.width,r.height);
                int x1=Math.max(0,cx-side/2),y1=Math.max(0,cy-side/2),x2=Math.min(input.cols(),cx+side/2),y2=Math.min(input.rows(),cy+side/2);
                if(x2>x1&&y2>y1){Mat roi=new Mat(input,new Rect(x1,y1,x2-x1,y2-y1));cropped=roi.clone();roi.release();}else cropped=input.clone();
            }
            Mat resized=new Mat();Imgproc.resize(cropped,resized,new Size(224,224),0.0,0.0,Imgproc.INTER_AREA);
            Mat lab=new Mat();Imgproc.cvtColor(resized,lab,Imgproc.COLOR_BGR2Lab);
            List<Mat> channels=new ArrayList<>();Core.split(lab,channels);
            CLAHE clahe=Imgproc.createCLAHE(2.0,new Size(8.0,8.0));Mat enhanced=new Mat();clahe.apply(channels.get(0),enhanced);channels.get(0).release();channels.set(0,enhanced);
            Mat merged=new Mat();Core.merge(channels,merged);Mat processed=new Mat();Imgproc.cvtColor(merged,processed,Imgproc.COLOR_Lab2BGR);
            resized.release();lab.release();merged.release();for(Mat c:channels)c.release();clahe.collectGarbage();clahe.clear();return processed;
        } finally {gray.release();threshold.release();hierarchy.release();for(Mat c:contours)c.release();if(cropped!=null)cropped.release();}
    }
    private static ByteBuffer tensor(Mat bgr) {
        require(bgr.rows()==224&&bgr.cols()==224&&bgr.channels()==3,"Processed layout mismatch");byte[] p=new byte[224*224*3];bgr.get(0,0,p);
        ByteBuffer b=ByteBuffer.allocateDirect(224*224*3*4).order(ByteOrder.LITTLE_ENDIAN);
        for(int k=0;k<p.length;k+=3){b.putFloat((p[k+2]&255)/255.0f);b.putFloat((p[k+1]&255)/255.0f);b.putFloat((p[k]&255)/255.0f);}b.rewind();return b;
    }
    private static double infer(Interpreter model,ByteBuffer in,ByteBuffer out) { in.rewind();out.clear();long t=System.nanoTime();model.run(in,out);double ms=(System.nanoTime()-t)/1e6;out.rewind();return ms; }

    private void startRun() {
        button.setEnabled(false);
        new Thread(() -> {
            File out=new File(getFilesDir(),"fidelity_results/new_20261016_"+System.currentTimeMillis());out.mkdirs();
            try {
                require(ByteOrder.nativeOrder()==ByteOrder.LITTLE_ENDIAN,"This frozen tensor contract requires little endian");
                // The official Android AAR packages libopencv_java4.so.
                System.loadLibrary("opencv_java4");
                File manifest=safe("manifest.tsv");String manifestText=new String(read(manifest),"UTF-8");
                List<String[]> rows=new ArrayList<>();String[] lines=manifestText.split("\\r?\\n");Set<String> keys=new HashSet<>();
                for(int k=1;k<lines.length;k++)if(!lines[k].trim().isEmpty()){String[] r=lines[k].split("\t",-1);require(r.length==8,"Manifest column count");require(keys.add(r[0]+":"+r[1]),"Duplicate input key");rows.add(r);}
                require(rows.size()==218,"Frozen manifest must contain 218 inputs, got "+rows.size());
                String configText=new String(read(getAssets().open("runtime_config.json")),"UTF-8");JSONObject config=new JSONObject(configText);
                write(new File(out,"manifest.tsv"),read(manifest));write(new File(out,"runtime_config.json"),configText);
                JSONObject env=new JSONObject().put("record_type",RECORD).put("application_implementation","Independent native Java Activity; SDK tools build, no Gradle or AndroidX")
                    .put("manufacturer",Build.MANUFACTURER).put("model",Build.MODEL).put("hardware",Build.HARDWARE).put("supported_abis",new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)))
                    .put("android_release",Build.VERSION.RELEASE).put("sdk",Build.VERSION.SDK_INT).put("opencv",Core.VERSION).put("opencv_runtime",Core.getVersionString())
                    .put("tflite_runtime",TensorFlowLite.runtimeVersion()).put("threads",4).put("xnnpack_requested",true).put("warmup_inferences_per_model",5)
                    .put("input_manifest_sha256",sha(manifest)).put("APK_sha256",sha(new File(getApplicationInfo().sourceDir))).put("package",getPackageName()).put("N_images",rows.size())
                    .put("preflight_before",preflight()).put("new_latency_role","DIAGNOSTIC ONLY. Debug fidelity measurements; never replace July7 benchmark or end-to-end table.")
                    .put("fixed_tensor_contract","Little-endian float32 NHWC RGB [0,1]; ImageNet normalization inside graph")
                    .put("raw_arm_contract","BitmapFactory decode; original DrPipeline integer crop/INTER_AREA/LAB CLAHE; no offline JPEG re-encoding")
                    .put("encoding_matched_arm_contract","Same raw processing, followed by default OpenCV JPEG encode/decode for .jpg/.jpeg only; PNG retains pixels. Device OpenCV decoder is recorded separately from the retained desktop PIL decoder. No equivalence assumed.");
                write(new File(out,"environment.json"),env.toString(2));
                List<String> models=new ArrayList<>();Iterator<String> ck=config.keys();while(ck.hasNext())models.add(ck.next());Collections.sort(models);
                int modelIndex=0;
                for(String key:models) {
                    modelIndex++;JSONObject spec=config.getJSONObject(key);byte[] modelBytes=read(getAssets().open(spec.getString("asset")));
                    require(sha(modelBytes).equals(spec.getString("sha256")),"Model hash mismatch "+key);
                    Interpreter.Options options=new Interpreter.Options().setNumThreads(4).setUseXNNPACK(true);
                    try(Interpreter interpreter=new Interpreter(direct(modelBytes),options)) {
                        require(interpreter.getInputTensor(0).dataType()==DataType.FLOAT32&&Arrays.equals(interpreter.getInputTensor(0).shape(),new int[]{1,224,224,3}),"Input model contract mismatch");
                        require(interpreter.getOutputTensor(0).dataType()==DataType.FLOAT32&&interpreter.getOutputTensor(0).numBytes()==20,"Output contract mismatch");
                        ByteBuffer output=ByteBuffer.allocateDirect(20).order(ByteOrder.LITTLE_ENDIAN);
                        ByteBuffer warm=direct(read(safe(rows.get(0)[4])));for(int n=0;n<5;n++)infer(interpreter,warm,output);
                        try(BufferedWriter writer=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(out,key+"_paired_android.jsonl")),"UTF-8"))) {
                            int count=0;
                            for(String[] row:rows) {
                                count++;File raw=safe(row[3]);byte[] referenceBytes=read(safe(row[4]));
                                require(sha(raw).equals(row[5])&&sha(referenceBytes).equals(row[6]),"Input hash mismatch "+row[1]);
                                float[] reference=floats(referenceBytes);
                                for(String arm:new String[]{"fixed_tensor","raw_pipeline","raw_encoding_matched"}) {
                                    long totalStart=System.nanoTime();double decode=0,prep=0,convert=0;Mat processed=null;ByteBuffer input;
                                    if(arm.equals("fixed_tensor")){long t=System.nanoTime();input=direct(referenceBytes);convert=(System.nanoTime()-t)/1e6;}
                                    else{long t=System.nanoTime();Mat bgr=rawBgr(raw);decode=(System.nanoTime()-t)/1e6;t=System.nanoTime();processed=preprocess(bgr);bgr.release();
                                        if(arm.equals("raw_encoding_matched")&&(raw.getName().toLowerCase(Locale.US).endsWith(".jpg")||raw.getName().toLowerCase(Locale.US).endsWith(".jpeg"))){MatOfByte encoded=new MatOfByte();require(Imgcodecs.imencode(".jpg",processed,encoded),"JPEG encoding failed");Mat decoded=Imgcodecs.imdecode(encoded,Imgcodecs.IMREAD_COLOR);encoded.release();require(!decoded.empty(),"JPEG decoding failed");processed.release();processed=decoded;}
                                        prep=(System.nanoTime()-t)/1e6;t=System.nanoTime();input=tensor(processed);convert=(System.nanoTime()-t)/1e6;}
                                    double inference=infer(interpreter,input,output);long t=System.nanoTime();double[] logits=new double[5],p=new double[5];double max=-Double.MAX_VALUE;int top=0;
                                    for(int k=0;k<5;k++){logits[k]=output.getFloat();if(logits[k]>max){max=logits[k];top=k;}}
                                    double sum=0;for(int k=0;k<5;k++){p[k]=Math.exp(logits[k]-max);sum+=p[k];}for(int k=0;k<5;k++)p[k]/=sum;
                                    double risk=p[2]+p[3]+p[4],outputMs=(System.nanoTime()-t)/1e6,totalMs=(System.nanoTime()-totalStart)/1e6;
                                    byte[] inputBytes=bytes(input);float[] observed=floats(inputBytes);double maxDiff=0,sumDiff=0;
                                    for(int k=0;k<observed.length;k++){double diff=Math.abs((double)observed[k]-(double)reference[k]);maxDiff=Math.max(maxDiff,diff);sumDiff+=diff;}
                                    JSONObject decisions=new JSONObject(),thresholds=spec.getJSONObject("thresholds");Iterator<String> it=thresholds.keys();while(it.hasNext()){String rule=it.next();double threshold=thresholds.getDouble(rule);decisions.put(rule,new JSONObject().put("threshold",threshold).put("positive",risk>=threshold).put("margin",risk-threshold));}
                                    JSONObject record=new JSONObject().put("record_type",RECORD).put("dataset",row[0]).put("image_id",row[1]).put("label",Integer.parseInt(row[2])).put("selection_reason",row[7])
                                        .put("model",key).put("model_sha256",spec.getString("sha256")).put("arm",arm).put("logits",json(logits)).put("probabilities",json(p)).put("top1",top).put("referable_risk",risk).put("decisions",decisions)
                                        .put("input_tensor_sha256",sha(inputBytes)).put("reference_tensor_sha256",row[6]).put("max_abs_input_diff_vs_retained_tensor",maxDiff).put("mean_abs_input_diff_vs_retained_tensor",sumDiff/observed.length)
                                        .put("raw_sha256",row[5]).put("decode_ms",decode).put("preprocess_ms",prep).put("tensor_ms",convert).put("inference_ms",inference).put("output_ms",outputMs).put("total_ms",totalMs);
                                    writer.write(record.toString());writer.newLine();writer.flush();
                                    if(processed!=null){MatOfByte png=new MatOfByte();require(Imgcodecs.imencode(".png",processed,png),"PNG export failed");String folder=arm.equals("raw_pipeline")?"android_raw_processed":"android_encoding_matched_processed";write(new File(out,folder+"/"+row[0]+"/"+row[1]+".png"),png.toArray());png.release();processed.release();}
                                }
                                report("New paired fidelity "+modelIndex+"/"+models.size()+": "+key+"\n"+count+"/"+rows.size()+": "+row[0]+" / "+row[1]+"\nKeep app open. Debug times are diagnostic only.");
                            }
                        }
                    }
                }
                env.put("preflight_after",preflight());write(new File(out,"environment.json"),env.toString(2));
                write(new File(out,"COMPLETE.txt"),"New paired fidelity run completed. No training, threshold tuning or historical timing replacement.\n");
                report("Complete: "+out+"\nRetrieve with adb exec-out run-as "+getPackageName()+" tar -cf - -C files fidelity_results");
            } catch(Throwable error) {
                StringWriter trace=new StringWriter();error.printStackTrace(new PrintWriter(trace));
                try{write(new File(out,"FAILED.txt"),trace.toString());}catch(Exception ignored){}
                report("Stopped: "+error+"\nPartial records: "+out);
            } finally {runOnUiThread(() -> button.setEnabled(true));}
        },"DR-fidelity-run").start();
    }
}
