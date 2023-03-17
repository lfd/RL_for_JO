SELECT * FROM movie_info AS mi, movie_companies AS mc, aka_title AS at, company_type AS ct WHERE mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;