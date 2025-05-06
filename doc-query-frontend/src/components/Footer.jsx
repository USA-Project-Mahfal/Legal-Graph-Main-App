import { AlertTriangle } from "lucide-react";

export default function Footer({ isDarkMode }) {
  return (
    <footer
      className={`px-6 py-3 text-sm ${
        isDarkMode
          ? "bg-gray-800 border-gray-700 text-gray-400"
          : "bg-gray-100 border-gray-200 text-gray-500"
      } border-t`}
    >
      <div className="flex justify-between items-center">
        <div>Legal Intelligence &copy; 2025</div>
        <div className="flex items-center">
          <AlertTriangle className="h-4 w-4 mr-1 text-amber-500" />
          <span>For informational purposes only. Not legal advice.</span>
        </div>
      </div>
    </footer>
  );
}
