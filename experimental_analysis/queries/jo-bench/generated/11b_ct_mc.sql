SELECT * FROM company_type AS ct, movie_companies AS mc WHERE ct.kind = 'production companies' AND mc.note IS NULL AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id;