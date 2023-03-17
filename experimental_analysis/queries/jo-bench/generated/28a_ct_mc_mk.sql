SELECT * FROM movie_companies AS mc, movie_keyword AS mk, company_type AS ct WHERE mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;