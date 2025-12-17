import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const mockData = [
  { month: "Jan", scans: 12, detected: 2 },
  { month: "Feb", scans: 19, detected: 3 },
  { month: "Mar", scans: 15, detected: 1 },
  { month: "Apr", scans: 25, detected: 4 },
  { month: "May", scans: 22, detected: 2 },
  { month: "Jun", scans: 30, detected: 5 },
];

export const AnalyticsChart = () => {
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={mockData}>
          <defs>
            <linearGradient id="colorScans" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(175, 65%, 40%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(175, 65%, 40%)" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorDetected" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(0, 72%, 51%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(0, 72%, 51%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(210, 25%, 88%)" />
          <XAxis dataKey="month" stroke="hsl(215, 15%, 45%)" fontSize={12} />
          <YAxis stroke="hsl(215, 15%, 45%)" fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(0, 0%, 100%)",
              border: "1px solid hsl(210, 25%, 88%)",
              borderRadius: "8px",
              boxShadow: "0 4px 6px -1px hsl(215 25% 15% / 0.1)",
            }}
          />
          <Area
            type="monotone"
            dataKey="scans"
            stroke="hsl(175, 65%, 40%)"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorScans)"
            name="Total Scans"
          />
          <Area
            type="monotone"
            dataKey="detected"
            stroke="hsl(0, 72%, 51%)"
            strokeWidth={2}
            fillOpacity={1}
            fill="url(#colorDetected)"
            name="Tumors Detected"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
