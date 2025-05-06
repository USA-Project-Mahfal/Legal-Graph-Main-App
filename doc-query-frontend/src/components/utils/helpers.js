export const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + " B";
  else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  else return (bytes / (1024 * 1024)).toFixed(1) + " MB";
};

export const getFileIcon = (fileType) => {
  if (fileType.includes("pdf")) return "📄";
  if (fileType.includes("word") || fileType.includes("docx")) return "📝";
  return "📋";
};

export const generateSampleResponse = (query) => {
  const responses = [
    "Based on the retrieved documents, specifically Section 4.2 of the NDA Agreement v2.1, data sharing with third parties is permitted under the following conditions: (1) the third party must sign a comparable NDA with confidentiality terms at least as restrictive as NDA X, (2) written approval must be obtained prior to sharing, and (3) a record of all shared data must be maintained. The Third Party Data Policy further specifies that any such sharing requires a Data Processing Addendum when personal information is involved.",
    "According to the legal documents analyzed, the statute of limitations for filing this type of claim is 3 years from the date of discovery. However, the agreement specifically mentions a reduced period of 2 years for contractual disputes in Section 7.3.",
    "The contract does not explicitly address this scenario. While Clause 12.4 covers force majeure events, remote work arrangements due to public health emergencies are not specifically included. I recommend seeking clarification through a formal amendment to the agreement.",
  ];

  return responses[Math.floor(Math.random() * responses.length)];
};
