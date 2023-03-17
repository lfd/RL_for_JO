SELECT * FROM movie_companies AS mc, company_type AS ct, movie_info AS mi WHERE ct.kind = 'production companies' AND mc.note LIKE '%(theatrical)%' AND mc.note LIKE '%(France)%' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;