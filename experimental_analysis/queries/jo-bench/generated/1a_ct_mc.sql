SELECT * FROM movie_companies AS mc, company_type AS ct WHERE ct.kind = 'production companies' AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%' AND (mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%') AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;