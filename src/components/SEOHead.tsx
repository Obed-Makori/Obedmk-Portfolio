import { Helmet } from "react-helmet";
import image from "../assets/image.png"; 

interface SEOHeadProps {
  title: string;
  description: string;
  canonical?: string;
  type?: string;
  structuredData?: object;
}

const SEOHead = ({
  title,
  description,
  canonical,
  type = "website",
  structuredData,
}: SEOHeadProps) => {
  const baseUrl = "https://obedmk.me";
  const fullTitle = `${title} | Obed Makori - Data Analyst & BI Developer`;
  const fullCanonical = canonical ? `${baseUrl}${canonical}` : baseUrl;

  
  const imageUrl = `${baseUrl}${image.replace(/^\.{0,2}\//, "/")}`;

  
  const defaultStructuredData = {
    "@context": "https://schema.org",
    "@type": "Person",
    "name": "Obed Makori",
    "url": baseUrl,
    "jobTitle": "Data Analyst & BI Developer",
    "sameAs": [
      "https://www.linkedin.com/in/obed-makori/",
      "https://github.com/obed-makori",
      "https://twitter.com/obedmakori254",
    ],
  };

  return (
    <Helmet>
      {/* Primary Meta Tags */}
      <title>{fullTitle}</title>
      <meta name="title" content={fullTitle} />
      <meta name="description" content={description} />
      <link rel="canonical" href={fullCanonical} />

      {/* Open Graph / Facebook */}
      <meta property="og:type" content={type} />
      <meta property="og:url" content={fullCanonical} />
      <meta property="og:title" content={fullTitle} />
      <meta property="og:description" content={description} />
      <meta property="og:image" content={imageUrl} />
      <meta property="og:locale" content="en_US" />
      <meta property="og:site_name" content="Obed Makori - Data Analyst" />

      {/* Twitter */}
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:url" content={fullCanonical} />
      <meta name="twitter:title" content={fullTitle} />
      <meta name="twitter:description" content={description} />
      <meta name="twitter:image" content={imageUrl} />
      <meta name="twitter:creator" content="@makori_obed254" />

      {/* Structured Data */}
      <script type="application/ld+json">
        {JSON.stringify(structuredData || defaultStructuredData)}
      </script>

      {/* Security Headers */}
      <meta httpEquiv="X-Content-Type-Options" content="nosniff" />
      <meta httpEquiv="X-Frame-Options" content="SAMEORIGIN" />
      <meta httpEquiv="X-XSS-Protection" content="1; mode=block" />
      <meta name="referrer" content="strict-origin-when-cross-origin" />
    </Helmet>
  );
};

export default SEOHead;
