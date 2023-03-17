SELECT * FROM movie_info AS mi, aka_title AS at, movie_companies AS mc, movie_keyword AS mk, keyword AS k, company_type AS ct WHERE mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;